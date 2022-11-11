from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import pickle
import numpy as np

def get_arx_model(pinode, df_, retrain):
    if retrain:
        data = df_.drop(columns=['Case']).copy()

        for seq in pinode.validation_sequences:
            data.iloc[seq[0]:seq[1],:] = np.nan

        all_X = []
        Y = []

        for seq in pinode.train_sequences:
            # Put the data together, mostly as differences
            for i in range(12, seq[1]-seq[0]):
                if np.any(np.isnan(data.iloc[seq[0]+i-12: seq[0]+i, :])):
                    break
                X = data.iloc[seq[0]+i-12: seq[0]+i, :].copy()
                all_X.append(X.values.flatten().reshape(1,-1))
                Y.append(data.loc[data.index[seq[0]+i], ['T_272', 'T_273', 'T_274']])

        X = np.concatenate(all_X, axis=0)
        Y = np.array(Y)#.reshape(-1,1)

        model = LinearRegression(fit_intercept=False, positive=False)
        model = model.fit(X, Y)
        arx = model.coef_
        prediction = model.predict(X)

        print(f'R2:\t\t\t{r2_score(Y, prediction):.3f}')
        with open(os.path.join('data', 'building_arx_model.pkl'), 'wb') as f:
            pickle.dump(arx, f)
    else:
        with open(os.path.join('data', 'building_arx_model.pkl'), 'rb') as f:
            arx = pickle.load(f)

    return arx

def get_arx_errors(arx, df_, sequences, retrain):
    if retrain:
        data = df_.drop(columns=['Case']).copy()

        preds = np.empty((len(sequences), 288, 3))
        Y = np.empty((len(sequences), 288, 3))

        for k, seq in enumerate(sequences):
            df = data.iloc[seq[0]:seq[1], :].copy()
            for i in range(12, seq[1]-seq[0]):

                X = df.iloc[i-12: i, :].values.flatten().reshape(1,-1)
                preds[k,i-12,:] = X.dot(arx.T)

                Y[k,i-12,:] = data.loc[data.index[seq[0]+i], ['T_272', 'T_273', 'T_274']]
                df.loc[df.index[i], ['T_272', 'T_273', 'T_274']] = preds[k, i-12, :].copy()

        errors = np.mean(np.abs(preds - Y), axis=0).mean(axis=-1)
        
        with open(os.path.join('data', 'building_arx_errors.pkl'), 'wb') as f:
            pickle.dump(errors, f)

    else:
        with open(os.path.join('data', 'building_arx_errors.pkl'), 'rb') as f:
            errors = pickle.load(f)
            
    return errors

def get_arx_predictions(arx, data_, sequence):
    
    data = data_.copy().drop(columns=['Case'])
    preds = np.empty((1, sequence[1] - sequence[0] - 12, 3))
    df = data.iloc[sequence[0]:sequence[1], :].copy()
    for i in range(12, sequence[1]-sequence[0]):
        X = df.iloc[i-12: i, :].values.flatten().reshape(1,-1)
        preds[0,i-12,:] = X.dot(arx.T)
        df.loc[df.index[i], ['T_272', 'T_273', 'T_274']] = preds[0, i-12, :].copy()
        
    return preds