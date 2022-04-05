import numpy as np

import torch
import torch.nn
import torchmetrics
import torchvision.models

#import timm

import matplotlib.pyplot as plt

import wandb

from sklearn.metrics import ConfusionMatrixDisplay

import pytorch_lightning as pl

def get_untrained_resnet(n_classes):
    # 23M parameters
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, n_classes)
    return model

def get_supervised_resnet(n_classes):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, n_classes)
    return model

def get_dino_resnet(n_classes):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    model.fc = torch.nn.Linear(2048, n_classes)
    return model

def get_dino_transformer(n_classes):
    model = torch.nn.Sequential(
        torch.hub.load('facebookresearch/dino:main', 'dino_vitb8'),
        torch.nn.Linear(768, n_classes)
    )
    return model

def get_barlow_resnet(n_classes):
    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    model.fc = torch.nn.Linear(2048, n_classes)
    return model

"""
def get_wide_resnet(n_classes):
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    model = timm.create_model('wide_resnet50_2', pretrained=True)
    model.fc = torch.nn.Linear(2048, n_classes)
    return model

def get_paws_resnet(n_classes):
    # The following code will load the paws pretrained model
    fp = f"{dataset.get_data_filepath()}/paws_imgnt_10percent_300ep_finetuned.pth.tar"
    model = torchvision.models.resnet50()
    state_dict = torch.load(fp, map_location='cpu')['encoder']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    new_state_dict['fc.weight'] = new_state_dict['fc.fc2.weight']
    new_state_dict['fc.bias']   = new_state_dict['fc.fc2.bias']
    for k in ["fc.fc1.weight", "fc.fc1.bias", "fc.bn1.weight", "fc.bn1.bias", "fc.bn1.running_mean", "fc.bn1.running_var", "fc.bn1.num_batches_tracked", "fc.fc2.weight", "fc.fc2.bias"]:
        new_state_dict.pop(k, None)
    model.load_state_dict(new_state_dict)
    model.fc = torch.nn.Linear(2048, n_classes)
    return model
"""

# Needed to remove extra dimensions in model
class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze()#.long()

class Model(pl.LightningModule):
    def __init__(self, lr, backbone_key, n_classes, class_names, class_weights):
        super().__init__()

        self.lr = lr
        self.n_classes = n_classes
        self.class_names = class_names
        
        # Get the desired backbone
        if   backbone_key == "untrained":
            insert = get_untrained_resnet(n_classes)

        elif backbone_key == "supervised":
            insert = get_supervised_resnet(n_classes)

        elif backbone_key == "wide":
            insert = get_wide_resnet(n_classes)

        elif backbone_key == "barlow":
            insert = get_barlow_resnet(n_classes)

        elif backbone_key == "dino":
            insert = get_dino_resnet(n_classes)

        elif backbone_key == "dino-transformer":
            insert = get_dino_transformer(n_classes)
        
        elif backbone_key == "paws":
            insert = get_paws_resnet(n_classes)

        else:
            print(f"Backbone {backbone_key} is not recognised")

        """
        # Need to do this squeeze fix for every model
        self.model = torch.nn.Sequential(
            insert,
            #Squeeze()
        )
        """
        self.model = torch.nn.Sequential(
            insert,
            Squeeze()
        )

        # If freezing the extractor/backbone then do so here
        #for param in self.model.parameters():
        #    param.requires_grad = False

        # Add the last layer
        #self.model[0].fc = torch.nn.Linear(2048, self.n_classes)

        # We'll use cross entropy, weighted by class
        self.loss_computer = torch.nn.CrossEntropyLoss(weight=torch.HalfTensor(class_weights))

        # Aids in save and loading
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)#.squeeze()

    # Shape / convienience helpers

    def _get_output_key_as_tensor(self, outputs, key):
        # Concat operation will complain if we pass it empty tensors
        to_concat = []
        for i, item in enumerate([ o[key] for o in outputs ]):
            # Zero dimensional tensors need to be handled specially (these
            # are often the loss tensors)
            if len(item.size()) == 0:
                to_concat.append(item.unsqueeze(0))
            else:
                to_concat.append(item)
        return torch.cat(to_concat, dim=0)

    def _get_y_and_y_hat(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = self(x)
        return {
            'y': y.long(), 
            'y_hat': y_hat
        }

    # Logging helpers

    def _log_step_scalar(self, label, scalar):
        self.log(
            label, 
            scalar,               
            on_step=True, 
            on_epoch=False, 
            prog_bar=True, 
            logger=True, 
            sync_dist=True
        )

    def _log_epoch_scalar(self, label, scalar):
        self.log(
            label, 
            scalar,    
            on_step=False, 
            on_epoch=True, 
            prog_bar=False, 
            logger=True, 
            sync_dist=True
        )

    # First the training routines

    def training_step(self, batch, batch_idx):
        # Calculate and log the step loss
        dct = self._get_y_and_y_hat(batch, batch_idx)

        # Compute, log and return loss
        loss = self.loss_computer(dct['y_hat'], dct['y'])
        self._log_step_scalar('train_loss_step', loss)
        return {
            'loss': loss
        }

    def training_step_end(self, outputs):
        # This needs to be passed through in parallel processing mode, otherwise
        # nothing will be sent to the validation epoch end function
        return outputs

    def training_epoch_end(self, outputs):
        # Calculate and log the epoch loss at the end of each training run
        avg_loss = self._get_output_key_as_tensor(outputs, 'loss').mean()
        self._log_epoch_scalar('train_loss_epoch', avg_loss)

    # Now the validation routines

    def validation_step(self, batch, batch_idx):
        return self._get_y_and_y_hat(batch, batch_idx)

    def validation_step_end(self, outputs):
        # This needs to be passed through in parallel processing mode, otherwise
        # nothing will be sent to the validation epoch end function
        return outputs

    def validation_epoch_end(self, outputs, log_label_prepend="val"):
        y     = self._get_output_key_as_tensor(outputs, 'y')
        y_hat = self._get_output_key_as_tensor(outputs, 'y_hat')

        # Calc and log loss
        loss = self.loss_computer(y_hat, y)
        self._log_epoch_scalar(f'{log_label_prepend}_loss_epoch', loss)

        # Calculate prec, recall, f1, acc
        prec = torchmetrics.functional.precision(y_hat, y)
        rec  = torchmetrics.functional.recall(y_hat, y)
        f1   = torchmetrics.functional.f1(y_hat, y)
        acc  = torchmetrics.functional.accuracy(y_hat, y)

        # Log it all
        self._log_epoch_scalar(f'{log_label_prepend}_prec_epoch', prec)
        self._log_epoch_scalar(f'{log_label_prepend}_rec_epoch',  rec)
        self._log_epoch_scalar(f'{log_label_prepend}_f1_epoch',   f1)
        self._log_epoch_scalar(f'{log_label_prepend}_acc_epoch',  acc)


        if log_label_prepend == "test":
            # I also want a confusion matrix
            preds = y_hat.max(1).indices # Get the max value logit / prediction

            # Use sklearn library for confusion matrix
            fig, ax = plt.subplots(figsize=(12, 12))
            #ax.set_xticks(range(19))
            #ax.set_yticks(range(19))
            cm_y_true = y.tolist()
            cm_y_pred = preds.tolist()
            cm_y_true.extend(list(range(self.n_classes)))
            cm_y_pred.extend(list(range(self.n_classes)))
            ConfusionMatrixDisplay.from_predictions(
                y_true=cm_y_true, 
                y_pred=cm_y_pred, 
                display_labels=self.class_names, 
                #normalize='all',
                xticks_rotation=90,
                #values_format=".2f",
                ax=ax
            )

            # Save it out, then log to wandb so its all 
            # in one place
            fp = f"{dutils.get_data_filepath()}/tmp-{dutils.get_rand_string(16)}.png"
            plt.savefig(fp)
            self.logger.experiment.log({F"{log_label_prepend}_confusion_matrix": wandb.Image(fp)})

    # Testing code parallels what we're doing with val

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)

    def test_epoch_end(self, outputs):
        # Same plots as val, but for test
        self.validation_epoch_end(outputs, log_label_prepend="test")

        """
        self.logger.experiment.log({
            "test_conf_mat" : wandb.plot.confusion_matrix(
                probs=y_hat.cpu().numpy(),
                y_true=y.cpu().numpy(), 
                preds=None, # Supplied probs, so don't need to supply
                class_names=self.class_names
            )
        })
        """

    # Optimisation functions

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)