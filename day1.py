ML basic flow yaad rakho:

Import → Load Data → Split → Train → Predict → Evaluate

## 📝 Machine Learning Notes: Overfitting, Underfitting, Regularization

---

### 1️⃣ Overfitting

* **Meaning:** Model training data ko sirf **memorize** kar leta hai, pattern nahi seekhta.
* **Signs:** Training accuracy high, test accuracy low
* **Example:** Exam ke liye sirf past papers yaad karna
* **Solutions:** More data, simpler model, regularization, dropout, early stopping
* **Trick:** "Memorize kar liya, generalize nahi"

---

### 2️⃣ Underfitting

* **Meaning:** Model **pattern bhi seekh nahi pata**, data ko poorly fit karta hai
* **Signs:** Training & test accuracy dono low
* **Example:** Complex data pe simple straight line fit karna
* **Solutions:** More complex model, more features, longer training, better algorithms
* **Trick:** "Samajh hi nahi paaya"

---

### 3️⃣ Balanced Model (Good Fit)

* **Goal:** Na simple na complex → generalizes well
* **Steps:**

  1. Right model complexity
  2. Zyada data
  3. Proper features
  4. Regularization (L1/L2)
  5. Train-test validation
  6. Early stopping
* **Example:** House price prediction with selected features
* **Trick:** "Na zyada yaad, na kam samajh, perfect fit"

---

### 4️⃣ Regularization

* **Purpose:** Overfitting se bachao by controlling weights

#### 🔹 L1 Regularization (Lasso)

* Formula: Loss_new = Loss_original + λ * sum(|w_i|)
* Effect: Some weights = 0 → automatic feature selection
* Use: Jab feature selection chahiye
* Trick: "I want Less features → zero weight"

#### 🔹 L2 Regularization (Ridge)

* Formula: Loss_new = Loss_original + λ * sum(w_i^2)
* Effect: Weights chhote ho jate hain, zero nahi
* Use: Sab features useful hain, bas control chahiye
* Trick: "Limit Large → small weights"

---

### 5️⃣ Programming Examples

#### Logistic Regression (scikit-learn)

```python
from sklearn.linear_model import LogisticRegression
model_l1 = LogisticRegression(penalty='l1', solver='liblinear')
model_l2 = LogisticRegression(penalty='l2', solver='liblinear')
```

#### Linear Regression (Ridge & Lasso)

```python
from sklearn.linear_model import Ridge, Lasso
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)
```

* `alpha` → regularization strength
* L1 → feature selection
* L2 → weight control

---

### 6️⃣ Real Life Use Cases

#### Feature Selection (L1)

* Medical diagnosis → select important tests
* E-commerce → select important user features
* Trick: "Backpack me sirf zaroori items"

#### Weight Control (L2)

* Finance/Stock → prevent single indicator over-influence
* Marketing → multiple features balanced
* Trick: "Backpack items evenly distribute"

---

### 7️⃣ Quick Tricks to Remember

* **Overfitting:** Memorized, fails on new data
* **Underfitting:** Poor understanding
* **Balanced:** Perfect fit, generalizes well
* **L1:** Less features, zero weights
* **L2:** All features, small weights
