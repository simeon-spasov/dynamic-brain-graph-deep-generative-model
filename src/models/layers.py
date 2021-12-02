import torch
import torch.nn as nn


def get_activation(name):
    """..."""

    activ_dict = {"relu": nn.ReLU(inplace=True), "lrelu": nn.LeakyReLU(inplace=True), "softplus": nn.Softplus(),
                  "elu": nn.ELU(inplace=True), "gelu": nn.GELU(), "selu": nn.SELU(inplace=True),
                  "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), None: nn.Identity()}
    
    if name not in activ_dict.keys():
        raise ValueError("activation function {} is not supported".format(name))
    
    else:
        return activ_dict[name] 


class LinearBlock(nn.Module):
    """..."""
    
    def __init__(self, in_features, out_features, activation="relu", batch_norm=False, dropout=0):
        
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=False if batch_norm else True)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.activation = get_activation(activation) 


    def forward(self, X):
        
        H = self.linear(X)

        if self.batch_norm is not None:
            H = self.batch_norm(H.transpose(-2, -1)).transpose(-2, -1)

        H = self.activation(H)

        if self.dropout is not None:
            H = self.dropout(H)

        return H


class GraphConv(nn.Module):
    """Graph convolution layer for undirected graph"""

    def __init__(self, in_features, out_features, bias=True):
        
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.b = nn.Parameter(torch.empty(out_features)) if bias else None
        self._reset_parameters()

    
    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.W)

        if self.b is not None: 
            nn.init.zeros_(self.b)

    
    def forward(self, X, A):
   
        H = X.matmul(self.W)
        H = A.bmm(H)

        if self.b is not None:
            H = H + self.b 
        
        return H

    
    def __repr__(self):

        msg = self.__class__.__name__ 
        msg += "(" + str(self.in_features) + ", " + str(self.out_features) 
        
        if self.b is None: 
            msg += ", bias=False" 
        
        msg += ")"

        return  msg


class GraphConvBlock(nn.Module):
    """..."""
    
    def __init__(self, in_features, out_features, activation=None, batch_norm=False, dropout=0):
        
        super().__init__()

        self.gconv = GraphConv(in_features, out_features, bias=False if batch_norm else True)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.activation = get_activation(activation) 

    def forward(self, X, A):

        if self.dropout is not None:
            X = self.dropout(X)
        
        H = self.gconv(X, A)

        if self.batch_norm is not None:
            H = self.batch_norm(H.transpose(-2, -1)).transpose(-2, -1)

        H = self.activation(H)

        return H


class GraphConvLSTMCell(nn.Module):
    """Graph convolution LSTM cell"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
   
        self._create_layers()
        self._reset_parameters()

    
    def _create_input_gate(self):
        self.conv_i = GraphConv(self.out_features, self.out_features, self.bias)
        self.weight_i = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.bias_i = nn.Parameter(torch.empty(self.out_features))

    
    def _create_forget_gate(self):
        self.conv_f = GraphConv(self.out_features, self.out_features, self.bias)
        self.weight_f = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.bias_f = nn.Parameter(torch.empty(self.out_features))

    
    def _create_cell_state(self):
        self.conv_c = GraphConv(self.out_features, self.out_features, self.bias)
        self.weight_c = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.bias_c = nn.Parameter(torch.empty(self.out_features))

    
    def _create_output_gate(self):
        self.conv_o = GraphConv(self.out_features, self.out_features, self.bias)
        self.weight_o = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.bias_o = nn.Parameter(torch.empty(self.out_features))

    
    def _create_layers(self):
        self._create_input_gate()
        self._create_forget_gate()
        self._create_cell_state()
        self._create_output_gate()

    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_i)
        nn.init.xavier_uniform_(self.weight_f)
        nn.init.xavier_uniform_(self.weight_c)
        nn.init.xavier_uniform_(self.weight_o)
        nn.init.zeros_(self.bias_i)
        nn.init.zeros_(self.bias_f)
        nn.init.zeros_(self.bias_c)
        nn.init.zeros_(self.bias_o)

        
    def _calculate_input_gate(self, X, A, hidden_state):
        input_gate = torch.matmul(X, self.weight_i)
        input_gate = input_gate + self.conv_i(hidden_state, A)
        input_gate = input_gate + self.bias_i
        input_gate = torch.sigmoid(input_gate)

        return input_gate

    
    def _calculate_forget_gate(self, X, A, hidden_state):
        forget_gate = torch.matmul(X, self.weight_f)
        forget_gate = forget_gate + self.conv_f(hidden_state, A)
        forget_gate = forget_gate + self.bias_f
        forget_gate = torch.sigmoid(forget_gate)
        return forget_gate

    
    def _calculate_cell_state(self, X, A, hidden_state, cell_state, input_gate, forget_gate):
        out = torch.matmul(X, self.weight_c)
        out = out + self.conv_c(hidden_state, A)
        out = out + self.bias_c
        out = torch.tanh(out)
        cell_state = forget_gate * cell_state + input_gate * out

        return cell_state

    
    def _calculate_output_gate(self, X, A, hidden_state):
        output_gate = torch.matmul(X, self.weight_o)
        output_gate = output_gate + self.conv_o(hidden_state, A)
        output_gate = output_gate + self.bias_o
        output_gate = torch.sigmoid(output_gate)

        return output_gate

    
    def _calculate_hidden_state(self, output_gate, cell_state):
        hidden_state = output_gate * torch.tanh(cell_state)

        return hidden_state

    
    def forward(self, X, A, hidden_state, cell_state):

        input_gate = self._calculate_input_gate(X, A, hidden_state)
        forget_gate = self._calculate_forget_gate(X, A, hidden_state)
        cell_state = self._calculate_cell_state(X, A, hidden_state, cell_state, input_gate, forget_gate)
        output_gate = self._calculate_output_gate(X, A, hidden_state)
        hidden_state = self._calculate_hidden_state(output_gate, cell_state)
        
        return hidden_state, cell_state



