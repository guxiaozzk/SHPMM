import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import timm
from typing import Any
torch.set_printoptions(threshold=np.inf)
class FocalLoss(nn.Module):
    def __init__(self, gamma = 1, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
# # 用于不均衡
# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in 
#         Focal Loss for Dense Object Detection.

#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

#         The losses are averaged across observations for each minibatch.

#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.


#     """
#     def __init__(self, class_num, alpha=None, gamma=0.7, size_average=True):
#         super(FocalLoss, self).__init__()
# #         if alpha is None:
# #             self.alpha = Variable(torch.ones(class_num, 1))
# #         else:
# #             if isinstance(alpha, Variable):
# #                 self.alpha = alpha
# #             else:
# #                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average

#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)

#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         #print(class_mask)


# #         if inputs.is_cuda and not self.alpha.is_cuda:
# #             self.alpha = self.alpha.cuda()
# #         alpha = self.alpha[ids.data.view(-1)]

#         probs = (P*class_mask).sum(1).view(-1,1)

#         log_p = probs.log()
#         #print('probs size= {}'.format(probs.size()))
#         #print(probs)
#         alpha=0.5
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
#         #print('-----bacth_loss------')
# #         print(batch_loss)


#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss
##for  baffine attention

class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
#         print(self.U.shape)
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
#         print('bilinar_mapping',bilinar_mapping.shape)
        return bilinar_mapping

class myModel(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dropout_rate: 0.0):
        super().__init__()
       
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*768, out_features=128),
                                            torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*768, out_features=128),
                                            torch.nn.ReLU())
        self.biaffne_layer = biaffine(128,1)

        self.lstm=torch.nn.LSTM(input_size=768,hidden_size=768, \
                        num_layers=1,batch_first=True, \
                        dropout=0.5,bidirectional=True)
        
        self.relu=torch.nn.ReLU()
#         self.logit = nn.Linear(7*7,768)
        self.logits_layer=torch.nn.Linear(in_features=768, out_features=2)
        

    def forward(self,input,inputs,is_training=False):
#         bert_output = self.roberta_encoder(input_ids=input['input_ids'], 
#                                             attention_mask=input['attention_mask'], 
#                                             token_type_ids=input['token_type_ids']) 
#         encoder_rep = bert_output[0]
#         bert_output2 = self.roberta_encoder(input_ids=inputs['input_ids'], 
#                                             attention_mask=inputs['attention_mask'], 
#                                             token_type_ids=inputs['token_type_ids']) 
#         encoder_rep2 = bert_output2[0]
        encoder_rep = input 
        encoder_rep2= inputs
#         print('encoder_rep',encoder_rep.shape)
        encoder_rep,_ = self.lstm(encoder_rep)
        
        encoder_rep2,_ = self.lstm(encoder_rep2)

        start_logits = self.start_layer(encoder_rep) 
        end_logits = self.end_layer(encoder_rep2) 
        # print("start_logits.shape",start_logits.shape)
        span_logits = self.biaffne_layer(start_logits,end_logits)
        # print("out of biaff",span_logits.shape) 
#         1,len,len,4
        span_logits = span_logits.reshape(span_logits.shape[0],span_logits.shape[1]*span_logits.shape[2]*span_logits.shape[3])
        ##形状不变
        # print("out contigupus",span_logits.shape)
        # span_logits = self.relu(span_logits)
        # span_logits = self.logits_layer(span_logits)

        span_prob = torch.nn.functional.softmax(span_logits, dim=-1)
        span_prob =  nn.Linear(span_prob.shape[1],768)(span_prob)
        ##全部拍成1 因为 拍成别的太大了  循环拍成1试试
        if is_training:
            return span_logits
        else:
            return span_prob

def attention_net2(lstm_output, final_state,l_tem,f_tem,mask=None,device=None):
#         print(lstm_output.shape,l_tem.shape)

        steps = lstm_output.shape[1]
        ##get baffine attention out
        baff_out = myModel(dropout_rate=0.0)(lstm_output.to('cpu'),lstm_output.to('cpu'))
#         print(baff_out.shape,lstm_output.shape)
#         out = torch.cat((lstm_out,baff_out.unsqueeze(1).to('cuda')),dim=1)
#         print(out.shape)
#         out = nn.Linear(out.shape[0], input_embeds.shape[1])
    
        #batch_size=1,n_step=length
        ##文本的Q和K是相同的
        # mask = np.triu(torch.ones(batch, n_q, n_k), k=1)
        # mask = torch.from_numpy(mask).byte()

        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        outputs=[]
        # attn_weights : [batch_size,n_step]
#         for i in range(steps):

#         if mask is not None:
#             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        lstm_out = lstm_output.transpose(1,2)
#             self.lstm_head()

        context = torch.bmm(lstm_out,soft_attn_weights.unsqueeze(2)).squeeze(2)
        

#         print(lstm_out.shape,context.unsqueeze(2).shape)  
#             1,768
#         batch_size = len(l_tem)
#         # hidden = final_state.view(batch_size,-1,1)
#         hidden = torch.cat((f_tem[0],f_tem[1]),dim=1).unsqueeze(2)
#         # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
#         attn_weights = torch.bmm(l_tem, hidden).squeeze(2)
#         outputs=[]
#         # attn_weights : [batch_size,n_step]
# #         for i in range(steps):

# #         if mask is not None:
# #             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

#         soft_attn_weights = F.softmax(attn_weights,1)

#         # context: [batch_size, n_hidden * num_directions(=2)]
#         l_tem = l_tem.transpose(1,2)
# #             self.lstm_head()

#         context_tem = torch.bmm(l_tem,soft_attn_weights.unsqueeze(2)).squeeze(2)
#         print('compare two shape',baff_out.shape,context.shape,lstm_out.shape)
        
        lstm_out =torch.cat((lstm_out,context.unsqueeze(2),baff_out.unsqueeze(2).to(device)),dim=2)
#         print('lstm_foratt',lstm_out.shape)
        mlp = nn.Linear(9,7).to(device)
#         print(type(lstm_out))
        context = mlp(lstm_out)
        context  = context.transpose(1,2).squeeze(0)
#             outputs.append(context)
        return context, soft_attn_weights

class PromptEncoder(nn.Module):
    '''learnable token generator modified from P-tuning
    https://github.com/THUDM/P-tuning
    '''
    def __init__(self, prompt_token_len, hidden_size, device, lstm_dropout,args,label_id_list):
        super().__init__()
        print("[#] Init prompt encoder...")
        # Input [0,1,2,3,4,5,6,7,8]
        self.seq_indices = torch.LongTensor(list(range(prompt_token_len))).to(device)
        # Embedding
        self.embedding = nn.Embedding(prompt_token_len, hidden_size)
        self.out_embedding = nn.Embedding(31112,hidden_size)
        # LSTM
        self.lstm_head = nn.LSTM(input_size=hidden_size,
                                       hidden_size=hidden_size // 2,
                                       num_layers=2,
                                       dropout=lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        
        self.mlp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size))
        self.device = device
        self.args = args
        self.label_id_list = label_id_list
        self.hidden_size=hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
#         self.mlp = nn.Linear(9,7).to(device)
    def attention_net2(self,lstm_output, final_state,l_tem,f_tem,mask=None,device=None):
#         print(lstm_output.shape,l_tem.shape)

        
        ##get baffine attention out
        
#         print(baff_out.shape,lstm_output.shape)
#         out = torch.cat((lstm_out,baff_out.unsqueeze(1).to('cuda')),dim=1)
#         print(out.shape)
#         out = nn.Linear(out.shape[0], input_embeds.shape[1])
    
        #batch_size=1,n_step=length
        ##文本的Q和K是相同的
        # mask = np.triu(torch.ones(batch, n_q, n_k), k=1)
        # mask = torch.from_numpy(mask).byte()
        # baff_out = myModel(dropout_rate=0.0)(lstm_output.to('cpu'),lstm_output.to('cpu'))
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # print(final_state.shape)
        lstm_output_o = lstm_output.clone()
        final_state_o = final_state.clone()
        steps = lstm_output.shape[1]
        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        outputs=[]
        # attn_weights : [batch_size,n_step]
#         for i in range(steps):

#         if mask is not None:
#             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        lstm_out = lstm_output.transpose(1,2)
#             self.lstm_head()

        context = torch.bmm(lstm_out,soft_attn_weights.unsqueeze(2)).squeeze(2)
        lstm_output,final_state = lstm_output[:,:9,:],final_state
        # print(final_state.shape)
        steps = lstm_output.shape[1]
        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        outputs=[]
        # attn_weights : [batch_size,n_step]
#         for i in range(steps):

#         if mask is not None:
#             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        lstm_out = lstm_output.transpose(1,2)
#             self.lstm_head()

        context_1 = torch.bmm(lstm_out,soft_attn_weights.unsqueeze(2)).squeeze(2)
        lstm_output,final_state =  lstm_output_o[:,9:,:],final_state_o
        # print(final_state.shape)
        steps = lstm_output.shape[1]
        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        outputs=[]
        # attn_weights : [batch_size,n_step]
#         for i in range(steps):

#         if mask is not None:
#             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        lstm_out = lstm_output.transpose(1,2)
#             self.lstm_head()

        context_2 = torch.bmm(lstm_out,soft_attn_weights.unsqueeze(2)).squeeze(2)
#         print(lstm_out.shape,context.unsqueeze(2).shape)  
#             1,768
#         batch_size = len(l_tem)
#         # hidden = final_state.view(batch_size,-1,1)
#         hidden = torch.cat((f_tem[0],f_tem[1]),dim=1).unsqueeze(2)
#         # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
#         attn_weights = torch.bmm(l_tem, hidden).squeeze(2)
#         outputs=[]
#         # attn_weights : [batch_size,n_step]
# #         for i in range(steps):

# #         if mask is not None:
# #             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

#         soft_attn_weights = F.softmax(attn_weights,1)

#         # context: [batch_size, n_hidden * num_directions(=2)]
#         l_tem = l_tem.transpose(1,2)
# #             self.lstm_head()

#         context_tem = torch.bmm(l_tem,soft_attn_weights.unsqueeze(2)).squeeze(2)
#         print('compare two shape',baff_out.shape,context.shape,lstm_out.shape)
        
        # lstm_out =torch.cat((lstm_out,context.unsqueeze(2),baff_out.unsqueeze(2).to(device)),dim=2)
        # print('lstm_beforatt',lstm_out.shape)
        # lstm_out1 = lstm_out
        steps = lstm_output_o.shape[1]
        lstm_out = lstm_output_o.transpose(1,2)
        # lstm_out =torch.cat((lstm_out,context_1.unsqueeze(2),context_2.unsqueeze(2)),dim=2)
        lstm_out =torch.cat((lstm_out,context.unsqueeze(2)),dim=2)

        # lstm_out =torch.cat((lstm_out,baff_out.unsqueeze(2).to(device)),dim=2)
        # print('lstm_foratt',lstm_out.shape)
        
        baff_out = myModel(dropout_rate=0.0)(lstm_out.transpose(1,2).to('cpu'),lstm_out.transpose(1,2).to('cpu'))
        # print(baff_out.shape)
        lstm_out =torch.cat((lstm_out,baff_out.unsqueeze(2).to(device)),dim=2)

        mlp = nn.Linear(lstm_out.shape[2],steps).to(device)
#         print(type(lstm_out))
        context = mlp(lstm_out)
        context  = context.transpose(1,2).squeeze(0)
#             outputs.append(context)
        return context, soft_attn_weights


    def forward(self):
#         print(input_embeds.shape)
        template = "How '' the sentence has '' emotion" 
        input_templatembeds = self.tokenizer(template,return_tensors='pt')['input_ids'].to(self.device)
#         print(input_templatembeds.shape[1])
        input_templates = self.out_embedding(input_templatembeds)
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        # print(h_n.shape)
        lstm_out,(h_n,c_n) = self.lstm_head(input_embeds)
        l_tem,(f_tem,c_tem) = self.lstm_head(input_templates)
#         print()
        lstm_out2 = self.attention_net2(lstm_out,h_n,l_tem,f_tem,device=self.device)[0]
#         print(lstm_out.shape,lstm_out2[0].shape)
        
#         output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        ##(1,7,7
        output_embeds = self.mlp_head(lstm_out).squeeze()
#         print(output_embeds.shape)
        return output_embeds


class VisualEncoder(nn.Module):
    def __init__(self, model_name, img_token_len, embedding_dim):
        super().__init__()
        self.is_resnet = False
        self.img_token_len = img_token_len
        self.embedding_dim = embedding_dim
        from timm.models.efficientnet import _cfg

        config = _cfg(url='', file='/home/zzk/snap/snapd-desktop-integration/57/Downloads/nf_resnet50_ra2-9f236009.pth') #file为本地文件路径

        self.backbone = timm.create_model(model_name,  pretrained=True,
#                                 features_only=True,
                                pretrained_cfg=config)
#         print(self.backbone.head)
        if "resnet" in model_name:
            self.is_resnet = True
            if model_name == "resnet50":
                self.global_pool = self.backbone.global_pool
            else:
                self.global_pool = self.backbone.head.global_pool
            self.visual_mlp = nn.Linear(2048, img_token_len * embedding_dim)  # 2048 -> n * 768
        elif "vit" in model_name:
            self.visual_mlp = nn.Linear(768, img_token_len * embedding_dim)  # 768 -> n * 768
        
    def forward(self, imgs_tensor):
        bs = imgs_tensor.shape[0]
        # print('img bs',bs)
        visual_embeds = self.backbone.forward_features(imgs_tensor)
        if self.is_resnet:
            visual_embeds = self.global_pool(visual_embeds).reshape(bs, 2048)
        # print(visual_embeds.shape)
#         with open('orginalimgrepresent.txt','w') as f:
# # # #                 print(it)
#                 f.writelines(str(visual_embeds))
        visual_embeds = self.visual_mlp(visual_embeds)
        visual_embeds = visual_embeds.reshape(bs, self.img_token_len, self.embedding_dim)
        # print(visual_embeds.shape)
        return visual_embeds


class MSAModel(torch.nn.Module):
    '''main model
    '''
    def __init__(self, args, label_id_list):
        super().__init__()
        self.args = args
        self.label_id_list = label_id_list

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
        self.lm_model = BertForMaskedLM.from_pretrained(args.model_name)

        self.embeddings = self.lm_model.bert.get_input_embeddings()
        self.embedding_dim = self.embeddings.embedding_dim  # 768
        self.para_forloss = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        if not args.no_img:
            self.img_token_id = self.tokenizer.get_vocab()[args.img_token]
            self.img_token_len = args.img_token_len
            self.visual_encoder = VisualEncoder(args.visual_model_name, self.img_token_len, self.embedding_dim)
            

        if args.template == 3:
            self.prompt_token_id = self.tokenizer.get_vocab()[args.prompt_token]
            self.prompt_token_len = sum([int(i) for i in args.prompt_shape.split('-')[0]]) + int(args.prompt_shape[-1])
            self.prompt_encoder = PromptEncoder(self.prompt_token_len, self.embedding_dim, args.device, args.lstm_dropout,args,self.label_id_list)

    def embed_input(self, input_ids, imgs=None):
        bs = input_ids.shape[0]
        embeds = self.embeddings(input_ids)

        if self.args.template == 3:
            prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
#             print('prompt_positiion',prompt_token_position)
            ##后几位不固定是因为句长不一样 获得到位置可以用来观察输出logit
            prompt_embeds = self.prompt_encoder()
            for bidx in range(bs):
                for i in range(self.prompt_token_len):
                    embeds[bidx, prompt_token_position[bidx, i], :] = prompt_embeds[i, :]
           
#         if self.args.template == 3:
#             print(input_ids.shape)
#             prompt_token_position = input_ids[:,:self.prompt_token_len+10-1].reshape((bs, self.prompt_token_len+10, 2))[:, :, 1]
#             prompt_embeds = PromptEncoder(self.prompt_token_len+10, self.embedding_dim, args.device, args.lstm_dropout)

#             for bidx in range(bs):
#                 for i in range(self.prompt_token_len+10):
#                     embeds[bidx, i, :] = prompt_embeds[i, :]
        
        
        if not self.args.no_img:
            visual_embeds = self.visual_encoder(imgs)
            # print("visual_embed",visual_embeds.shape)
#           
            img_token_position = torch.nonzero(input_ids == self.img_token_id).reshape((bs, self.img_token_len, 2))[:, :, 1]
            for bidx in range(bs):
                for i in range(self.img_token_len):
                    embeds[bidx, img_token_position[bidx, i], :] = visual_embeds[bidx, i, :]
        
        return embeds
    def get_top_n_presention(self,n,lis):
        global dic 
        dic = self.tokenizer.get_vocab()
        lis1 = sorted(lis,reverse=True)
        newlis=[]
        for i in lis1[:n]:
            newlis.extend( [k for k,v in dic.items() if v == lis.index(i)])
        return newlis
    def forward(self, input_ids, attention_mask, labels, imgs=None,method=None):
        # print(labels[labels!=-100])
        inputs_embeds = self.embed_input(input_ids, imgs)
        # print(input_ids.shape)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        output = self.lm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss, logits = output.loss, output.logits
        l_logits = []      
         #观察soft  prompt 位置的logit评分
        bs =  input_ids.shape[0]
        # print(labels[labels!=-100][0])
        # print("每个批次的大小 bs",bs)
        p_logits=[]
        # prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
        # #选一行position
        # prompt_token_position = prompt_token_position[0]

# #         print(logits.shape)
            
#         for id in range(len(prompt_token_position)):
#             p = logits[0,prompt_token_position[id],:]
# #             print((p))
# #             print(list(p))
# #             p = self.get_top_n_presention(5,list(p)
#             p_logits.append(p)
# #             f.writelines(str(p_logits))
# #         print(p_logits)
#         p_logits = torch.stack(p_logits)
#         p_logits = torch.tensor(p_logits)
        # print(list(p_logits.to('cpu')))

        
#         print(p_logits)
        img_token_position = torch.nonzero(input_ids == self.img_token_id).reshape((bs, self.img_token_len, 2))[:, :, 1]
        img_token_position = img_token_position
        
#         print(logits.shape)
        img_out=[]
        # print(img_token_position.shape)##32,3
        # print(img_token_position
        for it in range(len(img_token_position)):
            img_logits=[]
            for id in range(len(img_token_position[it])):
                p = logits[it,img_token_position[it,id],:]
    #             print((p))
    #             print(list(p))
    #             p = self.get_top_n_presention(5,list(p)
                img_logits.append(p)
#             f.writelines(str(p_logits))
#         print(p_logits)
            img_logits = torch.stack(img_logits)
            img_out.append(img_logits)
        img_out = torch.stack(img_out)
        # print('img_out',img_out.shape)
#         with open('imgrepresent.txt','w') as f:

#             for it in list(p_logits):
# #                 print(it)
#                 f.writelines(str(it))
        # print(logits.shape)
        logits = logits[labels != -100]
        # print(logits.shape)


        for label_id in range(len(self.label_id_list)):
            l_logits.append(logits[:, self.label_id_list[label_id]].unsqueeze(-1))
        l_logits = torch.cat(l_logits, -1)
#         print(l_logits.shape)
        probs = l_logits
#         print(probs)
        loss_func = FocalLoss()
#         print(logits.shape)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
#         print(logits.shape,labels[labels!=-100].shape)
       
#         print(labels)
        # print(labels[labels != -100])
#         print(inputs_embeds,labels,logits)
#         pred = logits[labels != -100]
#         print(pred.shape)
#         probs = pred[:, self.label_id_list]
#         print(probs,labels[labels != -100])
        lab = labels[labels != -100]
        # print(lab.shape)
        loss = loss_fct(logits.view(-1, self.lm_model.config.vocab_size),labels[labels!=-100].view(-1))
        le.fit(lab.to('cpu'))
#         mlp = nn.Linear(self.lm_model.config.vocab_size,3).to(self.args.device)
#         pred = mlp(pred)
#         print(lab)
        lab = torch.LongTensor(le.transform(lab.to('cpu')))
#         print(lab.shape)
#         loss=loss_fct(probs.to("cpu"),lab)  

#         loss=loss_func(torch.unsqueeze(probs,2).to("cpu"),torch.unsqueeze(lab ,1))  
        ##除了mask词 其余词损失为null
#         loss = loss_fct(probs.view(-1,3).to('cpu',lab)
#         print(probs.shape,labels.shape)
#         loss = loss_func(probs,labels)
        # probs_merge = probs[1::2]+probs[::2]
        probs_merge = probs
        pred_labels_idx = torch.argmax(probs_merge, dim=-1).tolist()
        y_ = [self.label_id_list[i] for i in pred_labels_idx]
        y = labels[labels != -100]
        # print(y.shape)
        # print(type(y),len(y.tolist()))
        
        # return loss, y_[::2], y.tolist()[::2],p_logits,img_out,probs_merge.cpu().tolist()

        # return loss, y_[1::2], y.tolist()[1::2],p_logits,img_out
        if method =='trai':
            return loss, y_[::2], y.tolist()[::2],p_logits,img_out,probs_merge.cpu().tolist()
        else:
            return loss, y_[::2], y.tolist()[::2],p_logits,img_out,probs_merge.cpu().tolist()
