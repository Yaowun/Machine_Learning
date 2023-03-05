import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import mnist


# read data
(data_train_x, data_train_y), (data_test_x, data_test_y) = mnist.load_data()
data_train_x = data_train_x.reshape(60000, 28, 28, 1)/255
data_test_x = data_test_x.reshape(10000, 28, 28, 1)/255
data_train_y = to_categorical(data_train_y)
data_test_y = to_categorical(data_test_y)
model_name = "model"

# file check
if not os.path.isdir(model_name):
    os.mkdir(model_name)
else:
    choice = input("Swap file '" + model_name + "' already exists!\n" 
                   "(U)se it, (O)verride it, (Q)uit:\n").lower()
    while True:
        if choice in "use":
            new_train = False
            save_model = False
            model = load_model(model_name)
            with open(model_name + "/history.pkl", "rb") as df:
                hist_df = pickle.load(df)
            break
        if choice in "override":
            new_train = True
            save_model = True
            break
        if choice in "quit":
            raise SystemExit
        else:
            sys.stdout.write("Please respond with 'use', 'override' or 'quit' " 
                             "(or 'u','o' or 'q').\n")

#%% model
if new_train:
    # create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))
    
    # set model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # train model
    batch_size = 32
    epochs = 150
    history = model.fit(data_train_x, 
                        data_train_y, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_split=0.1)

if save_model:
    model.save(model_name)
    hist_df = pd.DataFrame(history.history)
    with open(model_name + "/history.pkl", "wb") as df:
        pickle.dump(hist_df, df)

#%% example
fig = plt.figure(figsize=(5, 5))
plt.imshow(data_test_x[0], cmap='Greys')
fig.savefig(f"fig/{model_name}/number_example.png", bbox_inches="tight")
predict = model.predict(data_test_x)
print("Probability:\n", predict[0])
print("Predicted number is:", predict[0].argmax())

#%% evaluate model
score = model.evaluate(data_test_x, data_test_y, verbose=0)
print("Loss: {:.3f}".format(score[0]))
print("Accuracy: {:.2f}".format(score[1]*100)+" %")
fig = plt.figure(figsize=(12, 8))
plt.plot(hist_df["loss"])
plt.plot(hist_df["val_loss"])
plt.title("Train History", fontsize=30)
plt.ylabel("Loss", fontsize=26)
plt.xlabel("Epoch", fontsize=26)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.legend(["train", "validation"], fontsize=22)
plt.show()
fig.savefig(f"fig/{model_name}/train_history.png", bbox_inches="tight")
