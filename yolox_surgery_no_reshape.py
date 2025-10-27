import onnx
from ev_transforms.transforms import resize
from numba.cloudpickle import instance
from onnx import helper, numpy_helper, shape_inference, TensorProto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse

from sima_utils.onnx import onnx_helpers as oh
import numpy as np

def replace_slice_with_conv(model):
    """
    Replace Slice nodes with steps=2 with Conv nodes in an ONNX model.

    """
    graph = model.graph

    # Nodes to add and remove
    nodes_to_add = []
    nodes_to_remove = []
    initializer_to_add = []

    for i, node in enumerate(graph.node):
        if node.op_type == "Slice":
            starts = None
            ends = None
            axes = None
            steps = None

            if len(node.input) >= 5:
                steps_input_name = node.input[4]
                for initializer in graph.initializer:
                    if initializer.name == steps_input_name:
                        steps = numpy_helper.to_array(initializer)

                axes_input_name = node.input[3]
                for initializer in graph.initializer:
                    if initializer.name == axes_input_name:
                        axes = numpy_helper.to_array(initializer)

                starts_input_name = node.input[1]
                for initializer in graph.initializer:
                    if initializer.name == starts_input_name:
                        starts = numpy_helper.to_array(initializer)

            if steps is not None and np.array_equal(steps, [2]):
                print(f"Replacing Slice node {node.name} with Conv node")

                input_tensor = node.input[0]
                output_tensor = node.output[0]
                axis = axes[0] if axes is not None else 0

                # Create Conv node
                conv_kernel_size = [1,2] if axes==3 else [2,1]
                conv_weights = np.zeros((3, 3, 1, 2) if axes==3 else (3,3,2,1), dtype=np.float32)
                if starts == 0:
                    for c in range(3):
                        conv_weights[c, c, 0, 0] = 1
                elif starts == 1:
                    if axis == 2:
                        for c in range(3):
                            conv_weights[c, c, 1, 0] = 1
                    elif axis == 3:
                        for c in range(3):
                            conv_weights[c,c,0,1] = 1
                # conv_weights[:, :, 0, 1] = 0 if axes == 3 else

                # Create weight initializer for Conv
                weight_name = f"{node.name}_conv_weights"
                conv_weight_initializer = helper.make_tensor(
                    name=weight_name,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=conv_weights.shape,
                    vals=conv_weights.flatten().tolist(),
                )

                conv_node = helper.make_node(
                    "Conv",
                    inputs=[input_tensor, weight_name],
                    outputs=[output_tensor],
                    kernel_shape=conv_kernel_size,
                    strides=[1,2] if axes==3 else [2,1],
                    name=f"{node.name}_conv",
                )

                # Add the Conv node in place of the Slice node
                nodes_to_add.append((i, conv_node))
                initializer_to_add.append(conv_weight_initializer)
                nodes_to_remove.append(node)

    # Remove old Slice nodes and their inputs
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Add new Conv nodes and initializers
    for idx, conv_node in nodes_to_add:
        graph.node.insert(idx, conv_node)

    for initializer in initializer_to_add:
        graph.initializer.append(initializer)


def remove_node(model: onnx.ModelProto, node: Union[str, onnx.NodeProto], remove_only: bool = False):
    if isinstance(node, str):
        node = oh.find_node(model, node)

    if not remove_only:
        assert len(node.output) == 1
        is_last_node = any(node.output[0] == x.name for x in list(model.graph.output))
        true_input_to_removed_node = [name for name in node.input if not oh.is_initializer(model, name)]

        if is_last_node:
            connecting_node = oh.find_node_output(model, true_input_to_removed_node[0])
            connecting_node.output[0] = node.output[0]
        else:
            following_node, input_idx = oh.find_node_input(model, node.output[0])
            following_node.input[input_idx] = true_input_to_removed_node[0]

    model.graph.node.remove(node)

def add_initializer(model: onnx.ModelProto, initializer_name: str, initializer_value: np.ndarray, fp32=True):
    """
    Add an initializer to a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param initializer_name: Name of the initializer.
    :param initializer_value: Value of the initializer.
    """
    if fp32:
        model.graph.initializer.append(
            onnx.helper.make_tensor(
                name=initializer_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=initializer_value.shape,
                vals=initializer_value.flatten().tolist()
            )
        )
    else:
        model.graph.initializer.append(
            onnx.helper.make_tensor(
                name=initializer_name,
                data_type=onnx.TensorProto.INT64,
                dims=initializer_value.shape,
                vals=initializer_value.flatten().tolist()
            )
        )

def replace_reshape_with_split_concat(model: onnx.ModelProto, reshape_node_name: str, split_sizes1: np.ndarray, split_axis1: int, concat_axis1: int,
                                      split_sizes2: list = None, split_axis2: int = 0, concat_axis2: int = 0, two_pairs: bool = False):
    graph = model.graph

    reshape_node_idx = -1
    reshape_node = None
    for idx, node in enumerate(graph.node):
        if node.name == reshape_node_name:
            reshape_node = node
            reshape_node_idx = idx
            break

    if reshape_node is None:
        raise ValueError(f"Node with name {reshape_node_name} not found in the model.")

    input_name = reshape_node.input[0]
    final_output_name = reshape_node.output[0]

    graph.node.remove(reshape_node)

    split_sizes_tensor1 = numpy_helper.from_array(np.array(split_sizes1, dtype=np.int64),
                                                  name=reshape_node_name + "_split_sizes1")
    graph.initializer.extend([split_sizes_tensor1])

    if two_pairs:
        split_sizes_tensor2 = numpy_helper.from_array(np.array(split_sizes2, dtype=np.int64),
                                                      name=reshape_node_name + "_split_sizes2")
        graph.initializer.extend([split_sizes_tensor2])

    split_node_name1 = reshape_node_name + "_Split_1"
    split_output_names1 = [f"{split_node_name1}_out{i}" for i in range(len(split_sizes1))]
    split_node1 = helper.make_node(
        'Split',
        inputs=[input_name, split_sizes_tensor1.name],
        outputs=split_output_names1,
        name=split_node_name1,
        axis=split_axis1
    )

    concat_node_name1 = reshape_node_name + "_Concat_1"
    concat_output_name1 = concat_node_name1 + "_output" if two_pairs else final_output_name
    concat_node1 = helper.make_node(
        'Concat',
        inputs=split_output_names1,
        outputs=[concat_output_name1],
        name=concat_node_name1,
        axis=concat_axis1
    )

    if two_pairs:
        split_node_name2 = reshape_node_name + "_Split_2"
        split_output_names2 = [f"{split_node_name2}_out{i}" for i in range(len(split_sizes2))]
        split_node2 = helper.make_node(
            'Split',
            inputs=[concat_output_name1, split_sizes_tensor2.name],
            outputs=split_output_names2,
            name=split_node_name2,
            axis=split_axis2
        )

        concat_node_name2 = reshape_node_name + "_Concat_2"
        concat_output_name2 = final_output_name
        # concat_output_name2 =  concat_node_name2 + "_output"
        concat_node2 = helper.make_node(
            'Concat',
            inputs=split_output_names2,
            outputs=[concat_output_name2],
            name=concat_node_name2,
            axis=concat_axis2
        )

    graph.node.insert(reshape_node_idx, split_node1)
    graph.node.insert(reshape_node_idx + 1, concat_node1)

    # print("nodes inserted")

    if two_pairs:
        graph.node.insert(reshape_node_idx + 2, split_node2)
        graph.node.insert(reshape_node_idx + 3, concat_node2)

    # Ensure the connections to other nodes remain intact
    for node in graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == final_output_name:
                # node.input[idx] = (concat_node_name2 if two_pairs else concat_node_name1) + "_output"
                node.input[idx] = final_output_name

    return model

def main():
    parser = argparse.ArgumentParser(description="Optimize YOLOX ONNX model.")
    parser.add_argument(
        "--size",
        type=str,
        choices=["s", "m", "l", "x"],
        required=True,
        help="Specify the size of the YOLOX model."
    )
    args = parser.parse_args()

    size = args.size
    onnx_model_path = f"./yolox_{size}.onnx"
    output_model_path = f"./yolox_{size}_opt_no_reshapes.onnx"

    model = oh.load_model(onnx_model_path, load_only=True)

    # remove_node(model, oh.find_node_input(model, "images")[0])
    # remove_node(model, oh.find_node_input(model, "images")[0])
    # remove_node(model, oh.find_node_input(model, "images")[0])
    # remove_node(model, oh.find_node_input(model, "images")[0])
    # remove_node(model, oh.find_node_input(model, "images")[0])
    #
    # first_conv_node = oh.find_node_input(model, "images")[0]
    # first_conv_node.input[0] = "images_1"
    #
    # oh.remove_inputs_by_name_list(model, "images")
    # oh.add_input(model, "images_1", [1, 12, 320, 320])

    replace_slice_with_conv(model)

    remove_node(model, oh.find_node_output(model, "output"))

    last_conc_node = oh.find_node_output(model, "output")

    for i, input in enumerate(last_conc_node.input):

        last_res_node = oh.find_node_output(model, input)

        res_input = last_res_node.input[0]

        last_node_in_branch = oh.find_node_output(model, res_input)

        oh.add_output(model, f"output_{i}", [1, 85, int(20 * np.power(2, 2 - i)), int(20 * np.power(2, 2 - i))])
        last_node_in_branch.output[0] = f"output_{i}"

        remove_node(model, last_res_node, remove_only=True)
        # oh.remove_initializers(model, [last_res_node.input[1]])
        # add_initializer(model, last_res_node.input[1], np.int64([1, 85, -1, 1]), fp32=False)

        # replace_reshape_with_split_concat(model, last_res_node.name, split_sizes1=1*np.ones(int(20 * np.power(2, 2 - i))), split_axis1=2, concat_axis1=3)

    remove_node(model, last_conc_node, remove_only=True)
    oh.remove_outputs_by_name_list(model, ["output"])



    onnx.save_model(model, output_model_path)
    oh.remove_infer_shape(model)
    oh.save_model(model, output_model_path, save_only=False)

if __name__ == "__main__":
    main()
