# 3rd party imports
import numpy as np
import torch
import class_resolver
from torch_geometric.data import Dataset, Batch, Data
from toytrack import ParticleGun, Detector, EventGenerator

import sys
sys.path.append("../../../../")

from micro_challenges.micro_challenge_base import MicroChallenge
from src.graph_generation import build_edges, graph_intersection
from . import models

# Global definitions
device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12
model_resolver = class_resolver.Resolver(classes = [models.MLP], base = torch.nn.Module)

class BasicMetricLearningChallenge(MicroChallenge):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = BasicMetricLearningDataset
        self.network = model_resolver.lookup(hparams.get("network", None))(hparams)

    def get_input_data(self, batch):
        input_data = torch.stack(
            [batch[f"node_{feature}"] for feature in self.hparams["node_features"]], dim=-1
        ).float()
        input_data[input_data != input_data] = 0  # Replace NaNs with 0s

        return input_data

    def append_hnm_pairs(
        self, e_spatial, spatial, r_train=None, knn=None
    ):
        if r_train is None:
            r_train = self.hparams["r_train"]
        if knn is None:
            knn = self.hparams["knn"]

        knn_edges = build_edges(
            query=spatial,
            database=spatial,
            r_max=r_train,
            k_max=knn,
            remove_self_loops=True,
        )

        e_spatial = torch.cat([e_spatial, knn_edges], dim=-1)

        return e_spatial

    def append_random_pairs(self, e_spatial, spatial):
        n_random = int(self.hparams["randomisation"] * len(spatial))
        indices_src = torch.randint(
            0, len(spatial), (n_random,), device=self.device
        )
        indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
        random_pairs = torch.stack([indices_src, indices_dest])

        e_spatial = torch.cat(
            [e_spatial, random_pairs],
            dim=-1,
        )
        return e_spatial

    def append_signal_edges(self, batch, edges):
        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        true_edges = torch.cat(
            [batch.tracks_edge_index, batch.tracks_edge_index.flip(0)], dim=-1
        )

        edges = torch.cat(
            [edges, true_edges],
            dim=-1,
        )

        return edges

    def get_distances(self, embedding, pred_edges):
        reference = embedding[pred_edges[1]]
        neighbors = embedding[pred_edges[0]]

        try:  # This can be resource intensive, so we chunk it if it fails
            d = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            d = [
                torch.sum((ref - nei) ** 2, dim=-1)
                for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))
            ]
            d = torch.cat(d)

        return d

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        batch.edge_index, embedding = self.get_training_edges(batch)
        self.apply_embedding(batch, embedding, batch.edge_index)

        batch.edge_index, batch.y, batch.truth_map, true_edges = self.get_truth(
            batch, batch.edge_index
        )

        loss = self.loss_function(batch, embedding)

        self.log("train_loss", loss, batch_size=1)

        return loss

    def get_training_edges(self, batch):
        # Instantiate empty prediction edge list
        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        with torch.no_grad():
            embedding = self.apply_embedding(batch)

        # Append Hard Negative Mining (hnm) with KNN graph
        training_edges = self.append_hnm_pairs(
            training_edges, embedding
        )

        # Append random edges pairs (rp) for stability
        training_edges = self.append_random_pairs(
            training_edges, embedding
        )

        # Append true signal edges
        training_edges = self.append_signal_edges(batch, training_edges)

        # Remove duplicate edges
        training_edges = torch.unique(training_edges, dim=-1)

        return training_edges, embedding

    def get_truth(self, batch, pred_edges):
        # Calculate truth from intersection between Prediction graph and Truth graph
        true_edges = torch.cat(
            [batch.tracks_edge_index, batch.tracks_edge_index.flip(0)], dim=-1
        )

        pred_edges, truth, truth_map = graph_intersection(
            pred_edges,
            true_edges,
            return_y_pred=True,
            return_truth_to_pred=True,
            unique_pred=False,
        )

        return pred_edges, truth, truth_map, true_edges

    def apply_embedding(self, batch, embedding_inplace=None, training_edges=None):
        # Apply embedding to input data
        input_data = self.get_input_data(batch)
        if embedding_inplace is None or training_edges is None:
            return self(input_data)

        included_hits = training_edges.unique().long()
        embedding_inplace[included_hits] = self(input_data[included_hits])

    def loss_function(
        self, batch, embedding, pred_edges=None, truth=None
    ):
        if pred_edges is None:
            assert "edge_index" in batch.keys(), "Must provide pred_edges if not in batch"
            pred_edges = batch.edge_index

        if truth is None:
            assert "y" in batch.keys(), "Must provide truth if not in batch"
            truth = batch.y

        d = self.get_distances(embedding, pred_edges)
        if self.hparams.get("moated_loss", False):
            return self.moated_hinge_loss(truth, d)
        else:
            return self.hinge_loss(truth, d)

    def hinge_loss(self, truth, d):
        """
        Calculates the hinge loss

        Args:
            truth (``torch.tensor``, required): The truth tensor of composed of 0s and 1s, of shape (E,)
            d (``torch.tensor``, required): The distance tensor between nodes at edges[0] and edges[1] of shape (E,)
            square_loss (``bool``, optional): If True, use squared distance and margin. If False, use absolute distance and margin.
        Returns:
            ``torch.tensor`` The hinge loss mean as a tensor
        """

        negative_mask = truth == 0
        positive_mask = truth == 1

        margin = self.hparams["margin"] ** 2

        # Handle negative loss
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[negative_mask],
            torch.ones_like(d[negative_mask]) * -1,
            margin=margin,
            reduction="sum",
        )

        # Handle positive loss
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[positive_mask],
            torch.ones_like(d[positive_mask]),
            margin=margin,
            reduction="sum",
        )
        
        if self.hparams.get("balanced_loss", True):
            negative_loss = negative_loss / negative_mask.sum()
            positive_loss = positive_loss / positive_mask.sum()
            mean_loss = (negative_loss + self.hparams.get("pos_weight", 1.0) * positive_loss) / 2
        else:
            mean_loss = (negative_loss + self.hparams.get("pos_weight", 1.0) * positive_loss) / len(truth)

        return mean_loss

    def moated_hinge_loss(self, truth, d):
        """
        An augmented version of the hinge loss. Where the regular hinge loss has the following
        behavior:
        - For negative examples, the loss is 0 if d > margin, and margin - d otherwise
        - For positive examples, the loss is 0 if d = 0, and d otherwise
        This loss function has the following behavior:
        - For negative examples, the loss is 0 if d > margin, and margin - d otherwise
        - For positive examples, the loss is 0 if d < margin/2, and d - margin/2 otherwise
        """

        negative_mask = truth == 0
        positive_mask = truth == 1

        d = torch.sqrt(d + sqrt_eps)

        # Handle negative loss
        negative_loss = torch.stack(
            [
                ((self.hparams["margin"] + self.hparams.get("moat_margin", self.hparams["margin"]/10)) - d[negative_mask]),
                torch.zeros_like(d[negative_mask]),
            ], dim=-1
        ).max(dim=-1)[0].pow(2).sum()
 
        # Handle positive loss
        positive_loss = torch.stack(
            [
                (d[positive_mask] - (self.hparams["margin"] - self.hparams.get("moat_margin", self.hparams["margin"]/10))),
                torch.zeros_like(d[positive_mask]),
            ], dim=-1
        ).max(dim=-1)[0].pow(2).sum()

        if self.hparams.get("balanced_loss", True):
            negative_loss = negative_loss / negative_mask.sum()
            positive_loss = positive_loss / positive_mask.sum()
            mean_loss = (negative_loss + self.hparams.get("pos_weight", 1.0) * positive_loss) / 2
        else:
            mean_loss = (negative_loss + self.hparams.get("pos_weight", 1.0) * positive_loss) / len(truth)

        return mean_loss
        

    def shared_evaluation(self, batch):
        knn_num = 500 if "knn_val" not in self.hparams else self.hparams["knn_val"]
        knn_radius = self.hparams.get("knn_radius", self.hparams["r_train"])
        embedding = self.apply_embedding(batch)

        # Build whole KNN graph
        batch.edge_index = build_edges(
            query=embedding,
            database=embedding,
            r_max=knn_radius,
            k_max=knn_num,
            remove_self_loops=True,
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        batch.edge_index, batch.y, batch.truth_map, true_edges = self.get_truth(
            batch, batch.edge_index
        )

        d = self.get_distances(embedding, batch.edge_index)

        if self.hparams.get("moated_loss", False):
            loss = self.moated_hinge_loss(batch.y, d)
        else:
            loss = self.hinge_loss(batch.y, d)

        if hasattr(self, "trainer") and self.trainer.state.stage in [
            "train",
            "validate",
        ]:
            self.log_metrics(
                batch, loss, batch.edge_index, true_edges, batch.y
            )

        return {
            "loss": loss,
            "distances": d,
            "preds": embedding,
            "truth_graph": true_edges,
        }

    def log_metrics(self, batch, loss, pred_edges, true_edges, truth):
        true_pred_edges = pred_edges[:, truth == 1]

        total_eff = true_pred_edges.shape[1] / true_edges.shape[1]
        total_pur = true_pred_edges.shape[1] / pred_edges.shape[1]
        f1 = 2 * (total_eff * total_pur) / (total_eff + total_pur)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {
                "val_loss": loss,
                "lr": current_lr,
                "total_eff": total_eff,
                "total_pur": total_pur,
                "f1": f1,
            },
            batch_size=1,
        )

class BasicMetricLearningDataset(Dataset):

    def __init__(self, num_events=None, hparams=None, input_dir=None, data_name=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(transform, pre_transform, pre_filter)
        
        self.hparams = hparams
        self.particle_gun = ParticleGun(dimension=2, pt=(50), pphi=(-np.pi, np.pi), vx=0., vy=0.)

        # Initialize a detector
        self.detector = Detector(dimension=2).add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

        # Initialize an event generator
        self.event_generator = EventGenerator(self.particle_gun, self.detector, num_particles=10)
        
        self.events = [self.event_generator.generate_event() for _ in range(num_events)]
        self.convert_to_pyg()

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

    def collate(self, data_list):
        batch = Batch.from_data_list(data_list)