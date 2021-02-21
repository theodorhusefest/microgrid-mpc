import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as k


class LSTM:
    def __init__(
        self,
        datafile,
        target,
        features,
        n_in,
        n_out,
        lstm_nodes=60,
        train_ratio=0.8,
        initial_lr=0.1,
        scale=True,
        clean=False,
        sample_rate= 1,
    ):
        self.sample_rate = sample_rate
        self.data = self.load_data(datafile)
        self.target = target
        self.features = [target] + features
        self.n_vars = len(self.features)
        self.n_in = n_in
        self.n_out = n_out
        self.lstm_nodes = lstm_nodes
        self.train_ratio = train_ratio
        self.initial_lr = initial_lr
        self.scale = scale
        self.clean = clean

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.train_X, self.train_Y, self.test_X, self.test_Y = self.prepare_data()
        self.model = self.create_model()

    def prepare_data(self):
        """
        Preprocesses the data, transforms it to correct format and
        spilts into train and test sets.
        """

        df = self.data.filter(self.features)

        if self.clean:
            N = 10
            for column in df.columns:
                df[column] = np.convolve(df[column], np.ones(N) / N, mode="same")
        df = df.values

        if self.scale:
            df = self.scaler.fit_transform(df)

        values = self.series_to_supervised(df).values
        n_train = int(self.data.shape[0] * self.train_ratio)
        train = values[:n_train, :]
        test = values[n_train:, :]

        train_X, train_Y = (
            train[:, : self.n_vars * self.n_in],
            train[:, -self.n_vars * self.n_out :],
        )
        test_X, test_Y = (
            test[:, : self.n_vars * self.n_in],
            test[:, -self.n_vars * self.n_out :],
        )

        train_X = train_X.reshape((train_X.shape[0], self.n_in, self.n_vars))
        test_X = test_X.reshape((test_X.shape[0], self.n_in, self.n_vars))
        print("Prepared data with the following variables: {}".format(self.features))
        print(
            "Shape of train set {}, shape of test set {}.\n".format(
                train_X.shape, test_X.shape
            )
        )

        return (
            train_X.astype(np.float32),
            train_Y.astype(np.float32),
            test_X.astype(np.float32),
            test_Y.astype(np.float32),
        )

    def series_to_supervised(self, data, dropnan=True):
        """
        Transforms data to dataframe for training
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(self.n_in, 0, -1):
            cols.append(df.shift(i))
            names += [("var{}(t-{})".format(j + 1, i)) for j in range(n_vars)]
        for i in range(0, self.n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [("var{}(t)".format(j + 1)) for j in range(n_vars)]
            else:
                names += [("var{}(t+{})".format(j + 1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def create_model(self):
        """
        Sets up the keras LSTM model.
        """

        model = k.Sequential()
        model.add(
            k.layers.LSTM(
                self.lstm_nodes,
                input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
            )
        )
        model.add(k.layers.Dense(self.train_Y.shape[1]))

        opt = k.optimizers.Adam(learning_rate=self.initial_lr)

        model.compile(loss="mse", optimizer=opt)

        print(model.summary())
        return model

    def train_model(self, epochs=200, verbose=2):
        earlystopping = k.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True, verbose=True
        )
        reduceLR = k.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001, verbose=True
        )

        self.history = self.model.fit(
            self.train_X,
            self.train_Y,
            epochs=epochs,
            batch_size=64,
            callbacks=[earlystopping, reduceLR],
            validation_data=(self.test_X, self.test_Y),
            verbose=verbose,
            shuffle=False,
        )

        print(
            "\nTraining finished with loss = {} and validation loss = {}.\n".format(
                self.history.history["loss"][-1], self.history.history["val_loss"][-1]
            )
        )

    def load_data(self, datafile):
        data = pd.read_csv(datafile, parse_dates=["date"])
        data["time"] = data.date.dt.hour + data.date.dt.minute / 60
        return data[::self.sample_rate]

    def predict_on_test_set(self):
        """
        Predict and invert predictions
        Invert testset for plotting
        """
        yhat = self.model.predict(self.test_X)
        yhat = self.invert_scaling(yhat[:, :: self.n_vars])
        self.plot_X = self.invert_scaling(
            self.test_X[:, :, :: self.n_vars].reshape((self.test_X.shape[0], self.n_in))
        )
        return yhat

    def invert_scaling(self, X):
        X_max = self.scaler.data_max_[0]
        X_min = self.scaler.min_[0]

        return X * (X_max - X_min) - X_min

    def plot_test_set(self, yhat, num_steps, plot_every=1):
        num_steps = np.min([self.plot_X.shape[0], num_steps])
        plt.figure(figsize=(20, 10))
        for step in range(num_steps):
            if step % plot_every == 0:
                plt.plot(
                    range(
                        step,
                        step + self.n_out,
                    ),
                    yhat[step],
                    color="blue",
                    linestyle="dashed"
                )
            if step > num_steps:
                break

        plt.plot(range(num_steps), self.data[self.target].iloc[:num_steps], color="red", label="True")
        plt.title("LSTM Predictions")
        plt.legend()

    def plot_training_history(self):
        plt.figure()
        plt.plot(self.history.history["loss"][1:], label="train")
        plt.plot(self.history.history["val_loss"][1:], label="test")
        plt.legend()
