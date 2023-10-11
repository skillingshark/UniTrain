# UniTrain
UniTrain is an open-source, unified platform for effortless machine learning model training, evaluation, and deployment across diverse tasks. Experience seamless experimentation and model deployment with UniTrain.

## Installation Instructions
`pip --version `  
  
### If pip does not exist, then install pip using the following command:  

`sudo apt update`  
`sudo apt install python3-pip` 
 
  
### Install the *UniTrain* module using:  
`pip install -i https://test.pypi.org/simple/ UniTrain==0.2.3`    

### Install the *torch* module using:  
`pip install torch`  

*Note*: The above commands must run on Terminal or on Google Colab.  
*Note*: People running the above command on Colab must use a "!" before every command.  

## Run the following Python code to train the model

`import UniTrain
from UniTrain.utils.classification import get_data_loader, train_model
from UniTrain.models.classification import ResNet9
from UniTrain.utils.classification import parse_folder
import torch

if parse_folder("/content/data/"):
  train_dataloader = get_data_loader("/content/data/", 32, True, split='train')
  test_dataloader = get_data_loader("/content/data/", 32, True, split='test')

  model = ResNet9(num_classes=6)
  model.to(torch.device('cuda'))

  train_model(model, train_dataloader, test_dataloader,
              num_epochs=10, learning_rate=1e-3, checkpoint_dir='checkpoints',logger = "training.log", device=torch.device('cuda'))`


## Functions and Classes
### get_data_loader  
*get_data_loader(data_dir, batch_size, shuffle=True, transform = None, split='train')*    
  
Create and return a data loader for a custom dataset.      
*Args:*  
> data_dir (str): Path to the dataset directory.  
> batch_size (int): Batch size for the data loader.  
> shuffle (bool): Whether to shuffle the data (default is True).  

*Returns:*  
> DataLoader: PyTorch data loader.

  ### parse_folder
  *parse_folder(dataset_path)*
  
Create 
*Args:*
> dataset_path(str): Path to the directory which contains Data  

*Returns:*
> Return  

  ### train_model
  *train_model(model, train_data_loader, test_data_loader, num_epochs, learning_rate=0.001, checkpoint_dir='checkpoints', logger=None, device=torch.device('cpu')*
<br>    
Train a PyTorch model for a classification task.<br>  
*Args:*<br>  
> model (nn.Module): Torch model to train.  
> train_data_loader (DataLoader): Training data loader.  
> test_data_loader (DataLoader): Testing data loader.  
> num_epochs (int): Number of epochs to train the model for.  
> learning_rate (float): Learning rate for the optimizer.  
> checkpoint_dir (str): Directory to save model checkpoints.  
> logger (Logger): Logger to log training details.  
> device (torch.device): Device to run training on (GPU or CPU).  
  
*Returns:*  
> None

## evaluate_model
*evaluate_model(model, dataloader)*
<br>
Evaluate the model using evaluation dataset <br>
*Args:*<br>
> model: Your classification model
> dataloader: Dataloader that you will get from get_data_loader function
*Returns:*<br>


# DCGAN Model (Deep Convolutional Generative Adversarial Network)
## Functions and Classes
### get_data_loader  
*get_data_loader(data_dir, batch_size, shuffle=True, transform = None, split='train')*    
  
Create and return a data loader for a custom dataset.      
*Args:*  
> data_dir (str): Path to the dataset directory.  
> batch_size (int): Batch size for the data loader.  
> shuffle (bool): Whether to shuffle the data (default is True).  

*Returns:*  
> DataLoader: PyTorch data loader.
>
>  ### parse_folder
 *parse_folder(dataset_path)*

 
  
Create 
*Args:*
> dataset_path(str): Path to the directory which contains Data  

*Returns:*
> Return
>
> ### train_generator
> `train_generator(discriminator_model , generator_model, train_data_loader,  num_epochs, learning_rate=0.001, checkpoint_dir='checkpoints', logger=None, device=torch.device('cpu'))`
> 
> ### train_discriminator
> `train_discriminator(discriminator_model , generator_model, train_data_loader,  num_epochs, learning_rate=0.001, checkpoint_dir='checkpoints', logger=None, device=torch.device('cpu'))`
>
> ### train_model
> `train_model(model, train_data_loader, test_data_loader, num_epochs, learning_rate=0.001, checkpoint_dir='checkpoints', logger=None, device=torch.device('cpu')`
<br>    
Train a PyTorch model for a DCGAN task.<br>  
*Args:*<br>  
> model (nn.Module): Torch model to train.  
> train_data_loader (DataLoader): Training data loader.  
> test_data_loader (DataLoader): Testing data loader.  
> num_epochs (int): Number of epochs to train the model for.  
> learning_rate (float): Learning rate for the optimizer.  
> checkpoint_dir (str): Directory to save model checkpoints.  
> logger (Logger): Logger to log training details.  
> device (torch.device): Device to run training on (GPU or CPU).  
  
*Returns:*  
> None



  


`
