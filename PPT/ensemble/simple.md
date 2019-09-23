#
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

# 匯入資料
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# 三個基學習器
log_clf = LogisticRegression()
rf_clf = RandomForestClassifier()
svm_clf = SVC()

# 投票分類器
voting_clf = VotingClassifier( estimators=[("lr", log_clf), ("rf", rf_clf), ("svc", svm_clf)], voting="hard" )

# voting_clf.fit( X_train, y_train )
for clf in ( log_clf, rf_clf, svm_clf, voting_clf ):
  clf.fit( X_train, y_train )
  y_pred = clf.predict( X_test )
  print( clf.__class__.__name__, accuracy_score(y_test, y_pred) )
```
