from collections import OrderedDict

hidden_layer_specs = [5018, 1024, 205]
input_size = 25088
output_size = 102
hidden_layer_ordered_dict = OrderedDict()
for i in range(len(hidden_layer_specs)-1):
    test_val = len(hidden_layer_specs)-1
    if i == 0:
        in_size = input_size
    else:
        in_size = hidden_layer_specs[i]
    if i == len(hidden_layer_specs) - 1:
        out_size = output_size
    else:
        out_size = hidden_layer_specs[i]
    fc_name = 'fc{0}'.format(i)
    hidden_layer_ordered_dict[fc_name] = 'nn.Linear({0}, {1})'.format(in_size, out_size)
    relu_name = 'relu{0}'.format(i)
    last_relu_name = relu_name
    hidden_layer_ordered_dict[relu_name] = 'nn.ReLU()'
    dropout_name = 'dropout{0}'.format(i)
    last_dropout_name = dropout_name
    hidden_layer_ordered_dict[dropout_name] = 'nn.Dropout(p=0.5)'
del hidden_layer_ordered_dict[last_relu_name]
del hidden_layer_ordered_dict[last_dropout_name]
hidden_layer_ordered_dict['output'] = 'nn.LogSoftmax(dim=1)'
hidden_layer_ordered_dict