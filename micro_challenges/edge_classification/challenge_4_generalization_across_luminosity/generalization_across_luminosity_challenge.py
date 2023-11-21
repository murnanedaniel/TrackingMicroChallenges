# 3rd party imports
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import class_resolver
from torch_geometric.data import Dataset, Batch, Data
from toytrack import ParticleGun, Detector, EventGenerator
from tqdm import tqdm

import sys
import os
sys.path.append("../../../../")

from micro_challenges.micro_challenge_base import MicroChallenge
from src.graph_generation import build_edges, graph_intersection

# Global definitions
device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12

class GeneralizationAcrossLuminosityChallenge(MicroChallenge):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = GeneralizationAcrossLuminosityDataset
        # Get the absolute path to the models directory
        models_abs_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_resolver = self.get_model_resolver(models_abs_dir)
        self.network = self.model_resolver.lookup(hparams.get("network", None))(hparams)

    def setup(self, stage="fit"):
        print("Setting up the data...")
        if not self.trainset or not self.valset or not self.testset:
            for data_name, data_num, particle_num in zip(["trainset", "valset", "testset"], self.hparams["data_split"], self.hparams["num_particles"]):
                if data_num > 0:
                    input_dir = self.hparams["input_dir"] if "input_dir" in self.hparams else None
                    dataset = self.dataset_class(num_events=data_num, particle_num=particle_num, hparams=self.hparams, data_name = data_name, input_dir=input_dir)
                    setattr(self, data_name, dataset)
        
        try:
            self.logger.experiment.define_metric("val_loss", summary="min")
        except Exception:
            warnings.warn("Could not define metrics for W&B")

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss_function(output, batch)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        return loss

    def loss_function(self, output, batch):
        assert hasattr(batch, "y"), (
            "The batch does not have a truth label. Please ensure the batch has a `y`"
            " attribute."
        )

        if self.hparams.get("balanced_loss", False):
            negative_mask = (batch.y == 0)

            negative_loss = F.binary_cross_entropy_with_logits(
                output[negative_mask],
                torch.zeros_like(output[negative_mask]),
            )

            positive_mask = (batch.y == 1)
            positive_loss = F.binary_cross_entropy_with_logits(
                output[positive_mask],
                torch.ones_like(output[positive_mask]),
            )

            return positive_loss + negative_loss
        else:
            return F.binary_cross_entropy_with_logits(output, batch.y.float())

    def shared_evaluation(self, batch):
        output = self(batch)
        loss = self.loss_function(output, batch)
        batch.output = output.detach()
        self.log_metrics(batch, loss)


        return {
            "loss": loss.detach(),
            "all_truth": batch.y,
            "output": output.detach(),
            "batch": batch,
        }

    def log_metrics(self, batch, loss):
        preds = torch.sigmoid(batch.output) > self.hparams["edge_cut"]

        true = batch.y.sum().float()
        true_positive = (batch.y.bool() & preds).sum().float()
        positive = preds.sum().float()
        auc = roc_auc_score(
            batch.y.bool().cpu().detach(),
            torch.sigmoid(batch.output).float().cpu().detach(),
        )

        # Eff, pur, auc
        eff = true_positive / true
        pur = true_positive / positive
        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log_dict(
            {
                "current_lr": current_lr,
                "eff": eff,
                "pur": pur,
                "auc": auc,
                "val_loss": loss,
            },  # type: ignore
            sync_dist=True,
            batch_size=1,
            on_epoch=True,
            on_step=False,
        )

        return preds

class GeneralizationAcrossLuminosityDataset(Dataset):

    def __init__(self, num_events=None, particle_num=None, hparams=None, input_dir=None, data_name=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(transform, pre_transform, pre_filter)
        
        self.hparams = hparams
        self.particle_gun = ParticleGun(dimension=2, pt=(50), pphi=(-np.pi, np.pi), vx=0., vy=0.)

        # Initialize a detector
        self.detector = Detector(dimension=2).add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

        # Initialize an event generator
        self.event_generator = EventGenerator(self.particle_gun, self.detector, num_particles=particle_num)
        
        self.events = [self.event_generator.generate_event() for _ in range(num_events)]
        self.convert_to_pyg()
        self.build_graphs()

        print(f"Generated {len(self.events)} events")

    def len(self):
        return len(self.events)
    
    def get(self, idx):
        return self.events[idx]
    
    def convert_to_pyg(self):
        self.events = [self.convert_event(event) for event in self.events]

    def convert_event(self, event):
        return Data(
            num_nodes = event.hits.shape[0],
            node_x = torch.tensor(event.hits.x.values, dtype=torch.float),
            node_y = torch.tensor(event.hits.y.values, dtype=torch.float),
            tracks_edge_index = torch.tensor(event.tracks, dtype=torch.long),
            pid = torch.tensor(event.hits.particle_id.values, dtype=torch.long),
        )

    def build_graphs(self):
        for event in tqdm(self.events):
            event.edge_index = self.heuristic_edges(event)
            true_edges = torch.cat(
                [event.tracks_edge_index, event.tracks_edge_index.flip(0)], dim=-1
            )
            event.y, event.truth_map = graph_intersection(event.edge_index, true_edges, return_y_pred=True, return_truth_to_pred=True, unique_pred=True)
            
    def heuristic_edges(self, event, radius=0.1):
        # Setup the dimensions
        event.node_r = torch.sqrt(event.node_x**2 + event.node_y**2)
        event.node_phi = torch.atan2(event.node_y, event.node_x)
        
        # Build the edges, with r scaled down by 10
        edges = build_edges(query = torch.stack([event.node_r / 10, event.node_phi], dim=1).cuda(),
                            database = torch.stack([event.node_r / 10, event.node_phi], dim=1).cuda(),
                            r_max = radius,
                            k_max = 100,
                            remove_self_loops=True)
        
        # Duplicate edges in both directions and remove duplicates
        edges = torch.cat([edges, edges.flip(0)], dim=-1)
        edges = torch.unique(edges, dim=-1)

        return edges.cpu()