import tensorflow_datasets as tfds
# Adam is going to be the optimizer for both
from tensorflow.keras.optimizers import Adam
# Binary cross entropy is going to be the loss for both 
from tensorflow.keras.losses import BinaryCrossentropy
from generator_disc import build_generator, build_discriminator, FashionGAN

# Scale and return images only 
def scale_images(data): 
    image = data['image']
    return image / 255

def prepare_data(dataset_name='fashion_mnist'):
    # Reload the dataset 
    ds = tfds.load(dataset_name, split='train')
    # Running the dataset through the scale_images preprocessing step
    ds = ds.map(scale_images) 
    # Cache the dataset for that batch 
    ds = ds.cache()
    # Shuffle it up 
    ds = ds.shuffle(60000)
    # Batch into 128 images per sample
    ds = ds.batch(128)
    # Reduces the likelihood of bottlenecking 
    ds = ds.prefetch(64)
    return ds

def fashganModel():
    g_opt = Adam(learning_rate=0.0001) 
    d_opt = Adam(learning_rate=0.00001) 
    g_loss = BinaryCrossentropy()
    d_loss = BinaryCrossentropy()

    generator = build_generator()
    discriminator = build_discriminator()

    # Create instance of subclassed model
    fashgan = FashionGAN(generator, discriminator)
    # Compile the model
    fashgan.compile(g_opt, d_opt, g_loss, d_loss)
    return fashgan

