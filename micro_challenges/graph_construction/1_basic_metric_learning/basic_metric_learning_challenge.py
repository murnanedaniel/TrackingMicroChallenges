# 3rd party imports
import torch
import class_resolver
from torch_geometric.data import Dataset, Batch
from toytrack import ParticleGun, Detector, EventGenerator

from microchallenges import MicroChallenge
import models
0
# Global definitions
device = "cuda" if torch.cuda.is_available() else "cpu"
sqrt_eps = 1e-12
model_resolver = class_resolver.Resolver(classes = [models.], base = torch.nn.Module)

class BasicMetricLearningChallenge(MicroChallenge):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = BasicMetricLearningDataset
        self.network = model_resolver.lookup(hparams.get("network", None))(hparams)

class BasicMetricLearningDataset(Dataset):

    def __init__(self, num_events=None, hparams=None, input_dir=None, data_name=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(transform, pre_transform, pre_filter)
        
        self.hparams = hparams
        self.particle_gun = ParticleGun(dimension=2, pt=(5, 20), pphi=(-np.pi, np.pi), vx=(-0.1, 0.1), vy=(-0.1, 0.1))

        # Initialize a detector
        self.detector = Detector(dimension=2)
        self.detector.add_from_template('barrel', min_radius=0.5, max_radius=3, number_of_layers=10)

        # Initialize an event generator
        self.event_generator = EventGenerator(self.particle_gun, self.detector, num_particles=(30, 5, 'normal'))
        
        self.events = [self.event_generator.generate_event() for _ in range(num_events)]
        self.convert_to_graphs()

        print(f"Generated {len(self.events)} events")

    def len(self):
        return len(self.events)
    
    def get(self, idx):
        return self.events[idx]
    
    def collate(self, data_list):
        batch = Batch.from_data_list(data_list)