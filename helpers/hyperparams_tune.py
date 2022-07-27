from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
import lightgbm as lgb

class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result

    def lgb_binclass(self, para):
        reg = lgb.LGBMClassifier(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict_proba(self.x_test)[:,1]
        loss = -para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}
    
# from hyperopt import tpe, Trials
# from helpers.hyperparams_tune import HPOpt
# from hyperopt import hp

# lgb_binclass_params = {
#     'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.25)),
#     'max_depth': hp.choice('max_depth', np.arange(3, 17, 1, dtype=int)),
#     'max_bin': hp.choice('max_bin', np.arange(63, 255, 1, dtype=int)),
#     'subsample': hp.uniform('subsample', 0.8, 1),
#     'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
#     'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#     'is_unbalance': True,
#     'objective' : 'binary',  
#     'first_metric_only':True,
# }
# lgb_fit_params = {
#     'early_stopping_rounds': 5,
#     'verbose': False,
#     'eval_metric': 'auc',
#     'categorical_feature':list(range(cat_feats_end)),
# }
# lgb_para = dict()
# lgb_para['reg_params'] = lgb_binclass_params
# lgb_para['fit_params'] = lgb_fit_params
# lgb_para['loss_func' ] = lambda y, pred: roc_auc_score(y, pred)

# obj = HPOpt(X_train, X_test, y_train, y_test)

# lgb_opt = obj.process(fn_name='lgb_binclass', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)