import argparse
import onnx
popart_supported_onnx_op = {}

def check_popart_op_exist(onnx_path):
    model = onnx.load(onnx_path)
    op_set = []
    un_support_op = []
    for onnx_version in popart_supported_onnx_op:
        op_set.extend(popart_supported_onnx_op[onnx_version])
    op_set = set(op_set)
    print('our op set:', op_set)
    for node in model.graph.node: 
        if node.op_type not in op_set:
            un_support_op.append(node.op_type)
            print(node.op_type,' DO NOT SUPPORTED')
    return set(un_support_op)

def load_popart_suport_onnx_op(load_path):
    with open(load_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            op_name = line.split('-')[0]
            op_version = line.split('-')[1]
            if op_version not in popart_supported_onnx_op:
                popart_supported_onnx_op[op_version] =[]
                popart_supported_onnx_op[op_version].append(op_name)
            else:
                popart_supported_onnx_op[op_version].append(op_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('''use the following lines to export your model to onnx
     dummy_input=np.random.random((1,3,224,224))
    dummy_input=torch.tensor(dummy_input).float()
    torch.onnx.export(model, (dummy_input,), "tmp.onnx", verbose=True,opset_version=11)
    exit(1)''', add_help=False)
    parser.add_argument('--onnx-path', type=str, required=True, metavar="FILE", help='path to onnx file', )
    args, unparsed = parser.parse_known_args()
    load_popart_suport_onnx_op('./popart_supported_onnx_op.txt')
     
    check_popart_op_exist(args.onnx_path)