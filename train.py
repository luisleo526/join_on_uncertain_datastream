from model import IEJModel
from utils import generate_objects

if __name__ == '__main__':

    model = IEJModel(3, 4)
    objects, epsilon = generate_objects(10000, 3, 0.2)
