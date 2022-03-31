from torch import nn

class MLM(nn.Module):
        def __init__(self, args, plm, metadata, tokenizer, model_config, wrapper_class):
            super(MLM, self).__init__()
            self.args = args
            self.plm = plm
            self.tokenizer = tokenizer
            self.num_classes = len(metadata['classes'])
            head_name = [n for n,c in plm.named_children()][-1]
            plm_head = getattr(plm, head_name)
            max_loop = 5
            if not isinstance(plm_head, nn.Linear):
                module = plm_head
                found = False
                last_layer_full_name = []
                for i in range(max_loop):
                    last_layer_name = [n for n,c in module.named_children()][-1]
                    last_layer_full_name.append(last_layer_name)
                    parent_module = module
                    module = getattr(module, last_layer_name)
                    if isinstance(module, nn.Linear):
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"Can't not retrieve a linear layer in {max_loop} loop from the plm.")
                self.hidden_dims = self.original_head_last_layer.shape[-1]
            else:
                self.hidden_dims = self.head.weight.shape[-1]
                
            self.head = nn.Linear(self.hidden_dims, self.num_classes)
                  
        def forward(self, input):
            embd = self.bert(input)
            return self.head(embd)
