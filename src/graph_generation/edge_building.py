
import torch
import torch_geometric
from torch_geometric.nn import radius
from typing import Optional, Tuple, Union

def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    r_max: float = 1.0,
    k_max: int = 10,
    remove_self_loops: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Type hint
    
    edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max)

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    if remove_self_loops:
        edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list


def graph_intersection(
    input_pred_graph,
    input_truth_graph,
    return_y_pred=True,
    return_y_truth=False,
    return_pred_to_truth=False,
    return_truth_to_pred=False,
    unique_pred=True,
    unique_truth=True,
):
    """
    An updated version of the graph intersection function, which is around 25x faster than the
    Scipy implementation (on GPU). Takes a prediction graph and a truth graph, assumed to have unique entries.
    If unique_pred or unique_truth is False, the function will first find the unique entries in the input graphs, and return the updated edge lists.
    """

    if not unique_pred:
        input_pred_graph = torch.unique(input_pred_graph, dim=1)
    if not unique_truth:
        input_truth_graph = torch.unique(input_truth_graph, dim=1)

    unique_edges, inverse = torch.unique(
        torch.cat([input_pred_graph, input_truth_graph], dim=1),
        dim=1,
        sorted=False,
        return_inverse=True,
        return_counts=False,
    )

    inverse_pred_map = torch.ones_like(unique_edges[1]) * -1
    inverse_pred_map[inverse[: input_pred_graph.shape[1]]] = torch.arange(
        input_pred_graph.shape[1], device=input_pred_graph.device
    )

    inverse_truth_map = torch.ones_like(unique_edges[1]) * -1
    inverse_truth_map[inverse[input_pred_graph.shape[1] :]] = torch.arange(
        input_truth_graph.shape[1], device=input_truth_graph.device
    )

    pred_to_truth = inverse_truth_map[inverse][: input_pred_graph.shape[1]]
    truth_to_pred = inverse_pred_map[inverse][input_pred_graph.shape[1] :]

    return_tensors = []

    if not unique_pred:
        return_tensors.append(input_pred_graph)
    if not unique_truth:
        return_tensors.append(input_truth_graph)
    if return_y_pred:
        y_pred = pred_to_truth >= 0
        return_tensors.append(y_pred)
    if return_y_truth:
        y_truth = truth_to_pred >= 0
        return_tensors.append(y_truth)
    if return_pred_to_truth:
        return_tensors.append(pred_to_truth)
    if return_truth_to_pred:
        return_tensors.append(truth_to_pred)

    return return_tensors if len(return_tensors) > 1 else return_tensors[0]