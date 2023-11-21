import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import aggr

import sys
sys.path.append("../../../../../")
from src.ml_utils import make_mlp

class InteractionNetwork(nn.Module):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Define the dataset to be used, if not using the default

        self.hparams = hparams

        self.setup_aggregation()

        # Setup input network
        self.node_encoder = make_mlp(
            len(hparams["node_features"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # The edge network computes new edge features from connected nodes
        if self.hparams.get("edge_net_recurrent", True):
            self.edge_network = make_mlp(
                3 * hparams["hidden"],
                [hparams["hidden"]] * hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
            )
        else:
            self.edge_networks = nn.ModuleList(
                [
                    make_mlp(
                        3 * hparams["hidden"],
                        [hparams["hidden"]] * hparams["nb_edge_layer"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                    )
                    for _ in range(hparams["n_graph_iters"])
                ]
            )
            

        # The node network computes new node features
        if self.hparams.get("node_net_recurrent", True):
            self.node_network = make_mlp(
                self.network_input_size,
                [hparams["hidden"]] * hparams["nb_node_layer"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                output_activation=hparams["output_activation"],
                hidden_activation=hparams["hidden_activation"],
            )
        else:
            self.node_networks = nn.ModuleList(
                [
                    make_mlp(
                        self.network_input_size,
                        [hparams["hidden"]] * hparams["nb_node_layer"],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        output_activation=hparams["output_activation"],
                        hidden_activation=hparams["hidden_activation"],
                    )
                    for _ in range(hparams["n_graph_iters"])
                ]
            )            

        # Final edge output classification network
        self.output_edge_classifier = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )

    def message_step(self, x, start, end, e, i):
        # Compute new node features
        edge_messages = torch.cat(
            [
                self.aggregation(e, end, dim_size=x.shape[0]),
                self.aggregation(e, start, dim_size=x.shape[0]),
            ],
            dim=-1,
        )

        node_inputs = torch.cat([x, edge_messages], dim=-1)

        if self.hparams.get("node_net_recurrent", True):
            x_out = self.node_network(node_inputs)
        else:
            x_out = self.node_networks[i](node_inputs)            

        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        if self.hparams.get("edge_net_recurrent", True):
            e_out = self.edge_network(edge_inputs)
        else:
            e_out = self.edge_networks[i](edge_inputs)

        return x_out, e_out

    def output_step(self, x, start, end, e):
        classifier_inputs = torch.cat([x[start], x[end], e], dim=1)
        classifier_output = self.output_edge_classifier(classifier_inputs).squeeze(-1)


        return classifier_output

    def forward(self, batch, **kwargs):
        x = torch.stack(
            [batch[f"node_{feature}"] for feature in self.hparams["node_features"]], dim=-1
        ).float()
        start, end = batch.edge_index

        # Encode the graph features into the hidden space
        x.requires_grad = True
        if self.hparams.get("use_checkpoint", False):
            x = checkpoint(self.node_encoder, x, use_reentrant=False)
            e = checkpoint(
                self.edge_encoder, torch.cat([x[start], x[end]], dim=1), use_reentrant=False
            )
        else:
            x = self.node_encoder(x)
            e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            if self.hparams.get("use_checkpoint", False):
                x, e = checkpoint(
                    self.message_step, x, start, end, e, i, use_reentrant=False
                )
            else:
                x, e = self.message_step(x, start, end, e, i)

        return self.output_step(x, start, end, e)

    def setup_aggregation(self):
        if "aggregation" not in self.hparams:
            self.hparams["aggregation"] = ["sum"]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], str):
            self.hparams["aggregation"] = [self.hparams["aggregation"]]
            self.network_input_size = 3 * (self.hparams["hidden"])
        elif isinstance(self.hparams["aggregation"], list):
            self.network_input_size = (1 + 2 * len(self.hparams["aggregation"])) * (
                self.hparams["hidden"]
            )
        else:
            raise ValueError("Unknown aggregation type")

        try:
            self.aggregation = aggr.MultiAggregation(
                self.hparams["aggregation"], mode="cat"
            )
        except ValueError:
            raise ValueError(
                "Unknown aggregation type. Did you know that the latest version of"
                " GNN4ITk accepts any list of aggregations? E.g. [sum, mean], [max,"
                " min, std], etc."
            )