import warnings
warnings.filterwarnings('ignore')

from Trainer import Trainer
from Predictor import Predictor
from Preprocessor import Preprocessor


preprocessor = Preprocessor()
trainer =  Trainer()
predictor = Predictor()

train = preprocessor.get_train_dataset()
test = preprocessor.get_test_dataset()

"""
trainer.run(train)
"""

predictor.run(test)
