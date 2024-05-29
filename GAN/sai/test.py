import tensorflow as tf
from generator_disc import build_generator, build_discriminator, FashionGAN, ModelMonitor
import matplotlib.pyplot as plt

generator = build_generator()
# discriminator = build_discriminator()
# Load the trained GAN model
# gan = FashionGAN(generator, discriminator)

# Load the saved weights of the trained generator
generator.load_weights('generator_model.h5')  # Replace with your generator weights file path

# Generate synthetic images
num_images_to_generate = 16  # Specify the number of synthetic images you want to generate
latent_dim = 128  # Specify the dimensionality of the latent space

# Generate random latent vectors
random_latent_vectors = tf.random.normal(shape=(num_images_to_generate, latent_dim))

# Generate synthetic images using the generator
generated_images = generator.predict(random_latent_vectors)

# Ensure pixel values are in the range [0, 255]
generated_images = ((generated_images + 1) / 2.0) * 255.0

# Convert the images to uint8 format
generated_images = generated_images.astype('uint8')

# Display or save the synthetic images
for i in range(num_images_to_generate):
    # Display the synthetic images
    plt.imshow(generated_images[i])
    plt.axis('off')
    plt.show()

    # Save the synthetic images
    tf.keras.preprocessing.image.save_img(f'generated_image_{i}.png', generated_images[i])
