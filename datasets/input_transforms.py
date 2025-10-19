from torchvision import transforms

IMAGE_MEAN = (0, 0, 0)
IMAGE_STD = (1.0, 1.0, 1.0)

class Ensure3Channels:
    def __call__(self, img):
        return img.convert('RGB')  

def create_transforms_tips(image_size):
    transform = transforms.Compose([
        Ensure3Channels(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return transform, target_transform