""" collection of various helper functions for running ACE"""

from multiprocessing import dummy as multiprocessing
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import models

# This list is used to store the latent activations
# or gradients of the layer specified by the user
layer_activations = []
def forward_hook(module, input, output):
  layer_activations.append(output.squeeze().detach().numpy())
def hook_backwards(module, grad_input, grad_output):
  layer_activations.append(grad_output[0].detach().numpy())

def make_model(model_name):
  """creates instance of PyTorch model pre-trained on ImageNet1K

  Args:
    model_name (_type_): name of the model to be loaded (needs
      to be a model recognized by PyTorch)

  Raises:
    ValueError: If model is not recognized

  Returns:
    PyTorch model
  """
  if model_name == 'resnet18':
    # best available weights for ResNet-18 on ImageNet-1k
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
  elif model_name == 'googlenet':
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
  else:
    raise ValueError('Invalid model name')
  
  model.to('cuda' if torch.cuda.is_available() else 'cpu')

  return model

def get_layer_activations(model, layer:str, imgs:np.ndarray, 
                          batch_size:int=50, get_grads_for_cls_id:int=None):
  """Run input images through the PyTorch model and optain activations from
  specified layer

  Args:
    model: PyTorch model
    layer: layer within the model from which the activations are returned
    imgs: numpy-ndarray containing the imgs of shape [num_images, widht, height, channels=3]
    batch_size: batch size used for the calculations (reduces computational effort)
    get_gradients: instead of returning the activations of the specified layer, this parameter
      abuses the function to return the gradients w.r.t. a specific output class

  Returns:
    activations: NumPy array containing the model activations
  """

  if get_grads_for_cls_id is None:
    # register forward hook to get the activations from a specific layer
    hook_handle = model._modules.get(layer).register_forward_hook(forward_hook)
  else:
    # if get_grads_for_cls_id is set, instead register a backward hook which is used
    # to observe the gradients of a backward pass of the specified layer
    hook_handle = model._modules.get(layer).register_full_backward_hook(hook_backwards)
  # clear global list in which layer activations/gradients are stored
  del layer_activations[:]

  # Set the model to evaluation mode
  model.eval()

  # iterate over batches of images in imgs
  for i in tqdm(range(int(imgs.shape[0] / batch_size) + 1), desc='[INFO] calculating activations'):
    # convert images to a PyTorch tensor
    tensor_imgs = torch.tensor(imgs[i * batch_size:(i + 1) * batch_size].transpose((0, 3, 1, 2)), dtype=torch.float32)
    transformed_tensor_imgs = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)(tensor_imgs)
    transformed_tensor_imgs.to('cuda' if torch.cuda.is_available() else 'cpu')

    # forward pass through the model
    # if get_grads_for_cls_id=None the output is ignored and instead the activations of the 
    # specified layer are observed using a forward hook (see above)
    output = model(transformed_tensor_imgs)

    # if value for get_grads_for_cls_id is passed, calculate the gradients instead of the activations
    # for this, the cross-entropy loss is backpropagated through the network and the gradients are 
    # observed using a backward hook (see above)
    if not get_grads_for_cls_id is None:
      model.zero_grad()
      loss = torch.nn.functional.cross_entropy(
        output,
        torch.zeros(
          output.size(0), output.size(1)
          ).scatter_(
            1, torch.tensor([[get_grads_for_cls_id]] * output.size(0)), 1
            )
      )
      loss.backward()

  # remove the hook
  hook_handle.remove()

  # return the activations/gradients of the specified layer
  return np.concatenate(layer_activations, axis=0)

def create_dirs(working_dir:str, target_class:str, model_name:str, layer:str) -> str:
  """creates directories to load/save cached values and results

  Args:
      working_dir (str): argument passed by the user where the cached values and 
        results should be stored
      target_class (str): argmument passed by the user which class should be 
        interpreted
      model_name (str): argument passed by the user which model should be explained
      layer (str): argument passed by the user which layer should be used to calculate
        the activations

  Returns:
      str: base directory where cached values and results are loaded/saved
  """
  # create working directory if not already exists
  os.makedirs(working_dir, exist_ok=True)
  # create directory for model to be interpreted if not already exists
  os.makedirs(os.path.join(working_dir, model_name), exist_ok=True)
  # create directory for target class the be explained if not already exists
  os.makedirs(os.path.join(working_dir, model_name, target_class), exist_ok=True)
  # create directory for layer to calc acts from if not already exists
  base_dir = os.path.join(working_dir, model_name, target_class, layer)
  os.makedirs(base_dir, exist_ok=True)
  # create directories to load/save cached values and results
  os.makedirs(os.path.join(base_dir, 'acts'), exist_ok=True)
  print(f"[INFO] {os.path.join(os.getcwd(), base_dir)} is used to load/save cached values and results")
  return os.path.normpath(base_dir)

def load_image_from_file(filename:str, shape) -> np.array:
  """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """
  if not os.path.exists(filename):
    print('Cannot find file: {}'.format(filename))
    return None
  try:
    img = np.array(Image.open(filename).resize(shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255.0
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img
  except Exception as e:
    print(e)
    return None


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(299, 299),
                           num_workers=100):
  """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
  Returns:
    image arrays and succeeded filenames if return_filenames=True.
  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
  if run_parallel:
    pool = multiprocessing.Pool(num_workers)
    imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                    filenames[:max_imgs])
    if return_filenames:
      final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                         if imgs[i] is not None]
    imgs = [img for img in imgs if img is not None]
  else:
    for filename in filenames:
      img = load_image_from_file(filename, shape)
      if img is not None:
        imgs.append(img)
        if return_filenames:
          final_filenames.append(filename)
      if len(imgs) >= max_imgs:
        break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)

def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    Image.fromarray(image).save(address, format='PNG')

