from utils.all_utils import *
from utils.model import Perceptron

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

ETA = 0.3 # 0 and 1
EPOCHS = 10
model = Perceptron(eta=ETA, epochs=EPOCHS)

def create_dataset():
    df = pd.DataFrame(AND)
    X,y = prepare_data(df)
    return X,y

def create_model(X, y):    
    model.fit(X, y)
    _ = model.total_loss()
       

def model_predict(X):
    model.predict(X)

def load_model(path):
    model = joblib.load(path)
    return model    

def predict_load_model(inputs, path):
    model = load_model(path)
    model.predict(inputs)

def create_plot(file_name):
    df = pd.DataFrame(AND)
    save_plot(df, "or.png", model)
    

if __name__ == '__main__':
    model_path = "and.model"
    X, y = create_dataset()
    create_model(X, y)
    model_predict(X)
    save_model(model, model_path)
    #create_plot("plot_add")