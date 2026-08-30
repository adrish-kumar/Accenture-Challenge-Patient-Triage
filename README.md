# Accenture-Challenge-Patient-Triage
Automated ED Triage System — Technical Documentation Page 1
Automated Emergency Department Triage
System
Technical Documentation: Data, Model Architecture, Mathematics, Evaluation, and
Simulated Validation
This document describes the complete pipeline built to predict Emergency Department triage urgency (KTAS level)
from patient vitals, categorical intake data, and free-text chief complaints. It covers the source dataset, the Korean
Triage and Acuity Scale (KTAS) itself, every preprocessing and modeling step with the underlying mathematics, the
evaluation results actually produced by training, how prediction confidence and uncertainty are computed and
surfaced, and the 20-patient simulated validation set built to stress-test the system against ambiguous, pediatric,
geriatric, zero-history, surge, and clinician-override scenarios.
Report generated: August 30, 2026
Source files: KTAS_data_cleaned.xlsx · train_triage_model.py · triage_model.joblib · metrics_report.txt ·
Simulated_Triage_Patients_20.xlsx
Contents
1. Project Overview
2. What Is KTAS? The Korean Triage and Acuity Scale
3. Source Dataset: KTAS_data_cleaned.xlsx
4. Column-by-Column Data Dictionary
5. Features Used vs. Features Excluded (Leakage Prevention)
6. Preprocessing Pipeline & Mathematics
7. Model Architecture: Gradient Boosting & the Mathematics Behind It
8. Class Probabilities & Confidence Scoring
9. Model Selection: Comparing Three Candidate Models
10. Evaluation Results & Metric Definitions
11. Feature Importance
12. Simulated Validation Set (20 Patients)
13. System Files & Architecture
14. Limitations & Responsible Use
Automated ED Triage System — Technical Documentation Page 2
1. Project Overview
The goal of this project is to build a decision-support tool that predicts how urgently an Emergency Department (ED)
patient should be seen, using only information realistically available at the moment of triage: basic demographics,
vital signs, a handful of categorical intake fields, and the patient's own free-text description of why they came in (the
chief complaint). The target the model learns to predict is KTAS_expert — the retrospective, expert-assigned
gold-standard triage level in the source dataset.
The pipeline has four stages: (1) a public ED dataset is cleaned and structured; (2) a machine learning pipeline
combining numeric, categorical, and text features is trained and evaluated; (3) the trained model is packaged so it
can be queried for new patients with a full probability distribution over triage levels, not just a single answer; and (4)
the resulting system is stress-tested against a deliberately constructed set of 20 simulated patients covering
ambiguous presentations, pediatric and geriatric cases, a zero-history patient, a simulated 3× volume surge, and a
clinician override.
Throughout, this is treated as a decision-support prototype, not a certified medical device. Every prediction is
accompanied by an explicit confidence score, and low-confidence predictions are flagged for mandatory human review
rather than being silently accepted.
2. What Is KTAS? The Korean Triage and Acuity Scale
KTAS (Korean Triage and Acuity Scale) is a 5-level clinical triage system used in Korean emergency departments,
adapted from the Canadian Triage and Acuity Scale (CTAS). A trained triage nurse or physician assigns every
incoming patient a single integer level from 1 to 5 based on presenting complaint, vital signs, pain, and mental
status. The scale is inverted relative to everyday intuition: level 1 is the most life-threatening, and level 5 is the least
urgent.
KT
AS
Category Target
time-to-care
Description
1 Resuscitation Immediate Conditions that threaten life or limb requiring immediate aggressive
intervention (e.g. cardiac arrest, severe respiratory distress).
2 Emergent ≤ 15 minutes Conditions that are a potential threat to life, limb, or function,
requiring rapid medical intervention (e.g. chest pain suggestive of
ACS, severe shortness of breath, altered mental status).
3 Urgent ≤ 30 minutes Conditions that could progress to a serious problem requiring
emergency intervention; significant discomfort affecting function
(e.g. moderate abdominal pain, dehydration).
4 Less urgent ≤ 60 minutes Conditions related to patient age, distress, or potential for
deterioration that would benefit from intervention within 1-2 hours
(e.g. minor lacerations, uncomplicated fractures).
5 Non-urgent ≤ 120 minutes Conditions that may be acute but non-urgent, or part of a chronic
problem without evidence of deterioration (e.g. prescription refill,
mild rash).
Because level 1 is most urgent and level 5 is least urgent, an important asymmetry falls out of the numbering: if a
model predicts a higher KTAS number than the true level, it has under-triaged the patient (called them less urgent
than they actually are) — the clinically dangerous direction of error. Predicting a lower number than the truth is
Automated ED Triage System — Technical Documentation Page 3
over-triage — wasteful of resources, but safer. This asymmetry is used throughout this project's evaluation (Section
10).
The dataset distinguishes two triage labels for the same patient: KTAS_RN, the level assigned in real time by the
triage nurse, and KTAS_expert, a retrospective gold-standard level assigned later by reviewing the full case. The
model is trained to predict KTAS_expert — deliberately not KTAS_RN — because using the nurse's own
assessment as an input feature would mean the model is just parroting a human decision rather than making an
independent prediction (see Section 5).
3. Source Dataset: KTAS_data_cleaned.xlsx
The source file contains 1,267 real, de-identified Emergency Department patient records across 24 columns, drawn
from what the data indicates as two hospital sites (Group 1 and 2). Two data quality issues were fixed during initial
cleaning: the raw export had rows split across the wrong number of spreadsheet columns because commas inside
numeric fields (e.g. a European-style decimal '5,00') were incorrectly used to split columns; and a corrupted Excel
error token appeared throughout the NRS_pain column wherever pain was not applicable. Both were resolved by
reconstructing each row on its true delimiter and converting genuine error/blank markers into proper missing values.
3.1 Dataset-wide statistics
Statistic Value
Total patient records 1,267
Total columns 24
Sites (Group) Group 1: 688 patients • Group 2: 579 patients
Sex Female: 661 • Male: 606
Age Mean 54.4 years (SD 19.7) • range 16-96 • median 57
Missing: NRS_pain 556 of 1,267 (44%) — missing when Pain = 0 (no pain reported)
Missing: Saturation 697 of 1,267 (55%) — not measured for many lower-acuity visits
Missing: SBP / DBP / HR / RR / BT 18-29 records each (<3%)
Missing: Diagnosis in ED 2 of 1,267
3.2 KTAS_expert class distribution (the prediction target)
KTAS level Count % of dataset
1 – Resuscitation 26 2.1%
2 – Emergent 220 17.4%
3 – Urgent 487 38.4%
4 – Less urgent 459 36.2%
5 – Non-urgent 75 5.9%
This class imbalance matters: levels 1 and 5 together make up only 8% of the data, which is why the model performs noticeably
worse on those two classes (Section 10) — there simply are not many examples to learn from.
Automated ED Triage System — Technical Documentation Page 4
4. Column-by-Column Data Dictionary
All 24 original columns are listed below. Columns marked Used are fed into the model as predictive features.
Columns marked Excluded are present in the source data but deliberately withheld from the model (explained fully
in Section 5).
Column Type Role Description
Group Categorical Used Site/cohort code (1 or 2).
Sex Categorical Used 1 = Male, 2 = Female.
Age Numeric Used Age in years (16-96 in this dataset).
Patients number per
hour
Numeric Used ED arrival rate at the time of this patient's triage — a proxy for
how busy/surged the department is.
Arrival mode Categorical Used 1 = Walking in, 2 = 119 Ambulance (public EMS), 3 = Private
vehicle, 4 = Private ambulance, 5 = Wheelchair, 6 = Carried in,
7 = Other.
Injury Categorical Used 1 = No injury, 2 = Injury.
Chief_complain Free text Used Patient's own words describing why they came to the ED.
Converted to numeric features via TF-IDF (Section 6.3).
Mental Categorical Used AVPU-style mental status: 1 = Alert, 2 = Verbal response, 3 =
Pain response, 4 = Unresponsive.
Pain Categorical Used 0 = No pain reported, 1 = Pain reported.
NRS_pain Numeric Used Numeric Rating Scale for pain, 0-10. Missing when Pain = 0.
SBP Numeric Used Systolic blood pressure (mmHg).
DBP Numeric Used Diastolic blood pressure (mmHg).
HR Numeric Used Heart rate (beats per minute).
RR Numeric Used Respiratory rate (breaths per minute).
BT Numeric Used Body temperature (°C).
Saturation Numeric Used Oxygen saturation, SpO■ (%).
KTAS_RN Categorical
(1-5)
Excluded The triage nurse's own real-time triage decision. Excluded
because it is itself a human judgment call, not an independent
clinical fact — using it would make the model a
nurse-mimicking lookup, not an independent predictor.
Diagnosis in ED Free text Excluded Final diagnosis, only known after the full ED work-up is
complete — long after triage happens.
Disposition Categorical Excluded What happened to the patient at the end of the visit (e.g.
admitted, discharged). Known only after the visit ends.
KTAS_expert Categorical
(1-5)
TARGET Retrospective gold-standard triage level assigned by expert
review. This is what the model is trained to predict.
Error_group Categorical Excluded Derived retrospectively by comparing KTAS_RN against
KTAS_expert; cannot exist before both are known.
Automated ED Triage System — Technical Documentation Page 5
Column Type Role Description
Length of stay_min Numeric Excluded Total ED visit duration in minutes — only known after the visit
ends.
KTAS duration_min Numeric Excluded Time from arrival to KTAS assignment — a process-timing
artifact, not a clinical feature, and not available before triage.
mistriage Binary Excluded Derived retrospectively (KTAS_RN ≠ KTAS_expert); same
leakage issue as Error_group.
Automated ED Triage System — Technical Documentation Page 6
5. Features Used vs. Features Excluded (Leakage Prevention)
A model is only useful in production if it only ever sees, at prediction time, the information that would genuinely be
available at the moment of triage. Several columns in the source dataset describe things that are only known after
triage has already happened, or that are themselves a human triage decision. Including any of these as inputs would
produce artificially inflated accuracy during training/testing, followed by a model that cannot actually be used
prospectively — a classic case of data leakage.
5.1 The 16 features actually used
Feature type Columns
Numeric (9) Age, Patients number per hour, NRS_pain, SBP, DBP, HR, RR, BT, Saturation
Categorical (6) Group, Sex, Arrival mode, Injury, Mental, Pain
Text (1) Chief_complain (expanded into 300 TF-IDF features — see Section 6.3)
5.2 Columns deliberately excluded, and why
• KTAS_RN — a human triage decision. Feeding it into the model would mean the "prediction" is just repeating what
a nurse already decided, providing no independent value and no benefit in a scenario where the model is meant to
assist or check that decision.
• Diagnosis in ED, Disposition, Length of stay_min, KTAS duration_min — all only knowable after the ED visit is
complete or well underway. None of these exist at the moment a real patient walks up to the triage desk.
• Error_group, mistriage — both are derived after the fact by comparing KTAS_RN to KTAS_expert. They cannot
exist until both labels are already known, so they are unusable as predictive inputs by construction.
This exclusion policy is enforced directly in train_triage_model.py via the explicit NUMERIC_FEATURES,
CATEGORICAL_FEATURES, and TEXT_FEATURE lists, so the leakage-prone columns are never even loaded into the feature
matrix.
6. Preprocessing Pipeline & Mathematics
All three feature types are transformed independently by a ColumnTransformer, then concatenated into a single
feature vector before being handed to the classifier. The final feature vector has 328 dimensions: 9 numeric + 19
one-hot categorical (2+2+7+2+4+2 across the six categorical columns) + 300 TF-IDF text features.
6.1 Numeric features: median imputation + standardization
Missing numeric values (e.g. an unmeasured Saturation) are filled with that column's median from the training data,
chosen over the mean because vitals like blood pressure and pain score are not symmetric and the median is more
robust to outliers.
Every numeric feature is then standardized to zero mean and unit variance (a z-score), so that features on very
different scales (e.g. Age in years vs. HR in beats/min) contribute comparably during model fitting:
z = (x − µ) / σ
Automated ED Triage System — Technical Documentation Page 7
where x is the raw feature value, µ is that feature's mean over the training set, and σ is its standard deviation over the training
set.
6.2 Categorical features: mode imputation + one-hot encoding
Missing categorical values are filled with the training set's most frequent category. Each categorical column is then
one-hot encoded: a column with k possible categories becomes k binary indicator columns. For example, Arrival
mode (7 possible values) becomes 7 columns, each 0 or 1, with exactly one column equal to 1 per patient. Unknown
categories seen at prediction time (not present during training) are handled gracefully by encoding to all zeros rather
than raising an error.
6.3 Text feature: TF-IDF vectorization of the chief complaint
The free-text Chief_complain field is converted into 300 numeric features using TF-IDF (Term Frequency – Inverse
Document Frequency), restricted to unigrams and bigrams (single words and two-word phrases) that appear in at
least 2 patient records, with English stop words removed. For a term t in a document d, drawn from a corpus of n
documents:
tf(t, d) = number of times term t appears in document d
idf(t) = ln[ (1 + n) / (1 + df(t)) ] + 1
where df(t) is the number of documents (chief complaints) containing term t at least once. This is scikit-learn's
"smooth" IDF, which adds 1 to both numerator and denominator to avoid division by zero and prevent terms that
appear in every document from being given zero weight. The raw TF-IDF score is then:
tfidf(t, d) = tf(t, d) × idf(t)
Finally, each document's full TF-IDF vector is L2-normalized so that longer complaints are not automatically
weighted more heavily than shorter ones:
tfidf_norm(t, d) = tfidf(t, d) / √( Σ
t′
 tfidf(t′, d)2
 )
In practice, this means words that are common across almost every complaint ("pain", "the") get down-weighted, while
distinctive, diagnostically relevant terms ("chest", "dyspnea", "unresponsive") get up-weighted. Only the top 300 terms by
document frequency are kept, ranked as the actual model vocabulary at training time.
Automated ED Triage System — Technical Documentation Page 8
7. Model Architecture: Gradient Boosting & the Mathematics
Behind It
The final production model is a Gradient Boosting Classifier (scikit-learn's GradientBoostingClassifier), selected
after comparing three candidate algorithms by cross-validation (Section 9). Its configuration, exactly as trained:
Hyperparameter Value
n_estimators (boosting stages) 100
learning_rate (shrinkage, ν) 0.1
max_depth (per tree) 3
Number of classes (K) 5
Total input features after preprocessing 328
7.1 How gradient boosting works for multiclass classification
Gradient boosting builds an ensemble of shallow decision trees sequentially, where each new tree is trained to
correct the mistakes of the trees already added, rather than training many independent trees at once (as Random
Forest does). For a K-class problem, the model maintains one raw score function Fk
(x) per class k, and each
boosting stage m adds one small regression tree hk
(m) per class to each class's running score:
F
k
(m)(x) = F
k
(m−1)(x) + ν · h
k
(m)(x)
where ν is the learning rate (0.1 here) that shrinks each tree's contribution so that no single stage dominates and the
ensemble generalizes better. Each tree hk
(m) is fit to the negative gradient (pseudo-residual) of the loss function
with respect to Fk
 at the current stage — informally, "how wrong is the current model about class k for this patient,
and in which direction." The loss function used is multinomial deviance (equivalent to multiclass cross-entropy /
log-loss):
L = − Σ
k=1
K
 y
k
 · log p
k
(x)
where yk
 is 1 if the true class is k and 0 otherwise, and pk
(x) is the model's current predicted probability of class k,
obtained by passing the five raw class scores through a softmax function (Section 8). For this loss, the negative
gradient with respect to Fk
 works out to the simple, intuitive quantity:
residual
k
(x) = y
k
− p
k
(m−1)(x)
i.e. each new tree is literally trained to predict "how far off was my probability estimate for this class, and in which
direction" for every training patient, and the ensemble incrementally nudges its predictions toward the truth over all
100 stages.
Trees are kept shallow (max_depth = 3, meaning each tree makes at most 3 sequential splits) so that each one only
captures simple interactions between a handful of features at a time, relying on the boosting process — not any
single tree — to build up the full predictive relationship. This is a standard bias-variance trade-off: shallow trees are
individually weak learners, but hundreds of them, each correcting the last, compose into a strong model without
overfitting as readily as one very deep tree would.
8. Class Probabilities & Confidence Scoring
Automated ED Triage System — Technical Documentation Page 9
The model never returns a bare class label without also computing a full probability distribution across all five KTAS
levels. This is the mechanism by which the system "surfaces uncertainty" rather than silently returning a single
number.
8.1 From raw scores to probabilities: the softmax function
At prediction time, the trained ensemble produces one raw score Fk
(x) per KTAS class for a given patient x. These
five raw scores are converted into a valid probability distribution (five numbers between 0 and 1 that sum to exactly
1) using the softmax function:
p
k
(x) = exp(F
k
(x)) / Σ
j=1
5
 exp(Fj
(x))
This is exactly the vector reported as Prob_KTAS1 through Prob_KTAS5 in the simulated patient output (Section
12). The predicted class is simply the one with the highest probability:
Predicted_KTAS(x) = argmax
k
 p
k
(x)
8.2 Confidence and uncertainty banding
The system defines confidence as the probability assigned to the predicted class, expressed as a percentage:
Confidence(x) = 100 × max
k
 p
k
(x)
Confidence is then bucketed into three explicit bands so that a human reader never has to mentally interpret a raw
percentage under time pressure:
Band Confidence range System behavior
High ≥ 70% Prediction is used directly; no additional flag raised.
Moderate 50-69% Prediction is shown normally, but treated as advisory rather than
definitive.
Low < 50% Uncertainty_Flag is explicitly set to "Y – low confidence, escalate to
human review." The system never silently accepts a low-confidence
score.
This 50%/70% banding is a deliberate design choice, not something scikit-learn produces automatically: it is implemented
on top of the raw predict_proba output specifically so a confidence indicator is mandatory metadata on every single
prediction, satisfying the requirement that the system must never return a triage score without an accompanying
confidence value.
Automated ED Triage System — Technical Documentation Page 10
9. Model Selection: Comparing Three Candidate Models
Three classifiers were trained and compared using 5-fold stratified cross-validation on the training split (80% of the
data, with the remaining 20% held out untouched for final evaluation in Section 10). Cross-validation splits the
training data into 5 equal folds, trains on 4 of them, tests on the 5th, and repeats this 5 times so every fold is used as
the test set exactly once — the reported score is the average across all 5 runs, giving a much more reliable estimate
of real-world performance than a single train/test split.
The metric used for comparison is macro-F1 (defined precisely in Section 10.2), chosen because it weighs all five
KTAS classes equally regardless of how many patients fall into each one — important given the severe class
imbalance (Section 3.2), where using plain accuracy would let a model ignore the rare KTAS 1 and 5 classes entirely
and still score well.
Model 5-fold CV macro-F1 Configuration
Logistic Regression 0.548 class_weight='balanced', max_iter=2000
Random Forest 0.540 400 trees, class_weight='balanced_subsample'
Gradient Boosting 0.584 100 stages, learning_rate=0.1, max_depth=3 — selected
Gradient Boosting was selected as the best-performing model by this criterion and is the model refit on the full
dataset and shipped as triage_model.joblib.
10. Evaluation Results & Metric Definitions
All numbers in this section come directly from evaluating the selected model on the 20% held-out test set (254
patients) that the model never saw during training or cross-validation.
10.1 Headline results
Metric Value Meaning
Accuracy 0.709 Fraction of test patients where the predicted KTAS level exactly matched
the expert label.
Macro F1 0.566 Unweighted average of F1 across all 5 classes (Section 10.2).
Weighted F1 0.695 F1 averaged across classes, weighted by how many patients are in each
class.
Exact match rate 0.709 Identical to accuracy.
Within ±1 KTAS level 0.937 Fraction of predictions that were correct or off by at most one level in
either direction.
Under-triage rate 0.138 Fraction of predictions that were LESS urgent than the true level — the
clinically dangerous direction (Section 2).
Over-triage rate 0.154 Fraction of predictions that were MORE urgent than the true level.
10.2 How each metric is calculated
For a given class k, let TPk
, FPk
, and FNk
 be the number of true positives, false positives, and false negatives for
that class on the test set. Then:
Automated ED Triage System — Technical Documentation Page 11
Precision
k
 = TP
k
 / (TP
k
 + FP
k
) Recall
k
 = TP
k
 / (TP
k
 + FN
k
)
F1
k
 = 2 × (Precision
k
 × Recall
k
) / (Precision
k
 + Recall
k
)
Macro F1 = (1/K) Σ
k=1
K
 F1
k
 Weighted F1 = Σ
k=1
K
 (n
k
/N) · F1
k
where nk
 is the number of true patients in class k and N is the total test set size (254). For the clinical safety metrics,
let ∆
i
 = Predicted_KTASi
− True_KTASi
 for test patient i:
Under-triage rate = (1/N) Σ
i=1
N ■[∆
i
 > 0] Over-triage rate = (1/N) Σ
i=1
N ■[∆
i
 <
0]
where ■[·] is the indicator function (1 if the condition is true, 0 otherwise). Since higher KTAS numbers mean lower urgency, ∆
i
 >
0 means the model predicted a less urgent level than reality — an under-triage event.
10.3 Per-class performance
KTAS class Precision Recall F1-score Support (n)
1 (Resuscitation) 0.667 0.400 0.500 5
2 (Emergent) 0.658 0.568 0.610 44
3 (Urgent) 0.700 0.786 0.740 98
4 (Less urgent) 0.755 0.804 0.779 92
5 (Non-urgent) 0.400 0.133 0.200 15
The model performs best on the two largest, most common classes (3 and 4, together 74% of all patients) and
noticeably worse on the rarest classes (1 and 5, together only 8% of patients, with just 5 and 15 test examples
respectively). Recall of only 0.133 on class 5 means the model misses the majority of truly non-urgent patients,
tending to over-triage them into a more urgent bucket instead — a safe-but-inefficient failure mode. This is a direct,
expected consequence of how little training data exists for the rare classes, not a flaw specific to the algorithm
chosen.
10.4 Confusion matrix
Automated ED Triage System — Technical Documentation Page 12
Rows are the true expert KTAS level; columns are the model's prediction. The diagonal represents correct predictions. Most off-diagonal
mass sits immediately adjacent to the diagonal (level 2 mistaken for 3, level 4 mistaken for 3, etc.), consistent with the 93.7%
within-one-level accuracy reported above.
Automated ED Triage System — Technical Documentation Page 13
11. Feature Importance
Gradient Boosting reports feature importance based on total impurity reduction: for each feature j, its importance is
the sum, over every split in every tree in the ensemble that uses feature j, of the weighted decrease in node impurity
that the split achieved, averaged over all trees and normalized so all feature importances sum to 1:
Importance(j) = (1/M) Σ
trees
Σ
splits on j w
node
 · ∆impurity
where M is the number of trees, wnode is the fraction of training samples reaching that node, and ∆impurity is the
reduction in deviance achieved by that split. Intuitively: a feature is "important" if splitting on it consistently and
substantially improves the model's predictions across many trees.
Top 20 features by importance. The pain score (NRS_pain) and the presence of the word "chest" in the chief complaint are the two
strongest single predictors, followed by whether an injury is present and several vital signs (SBP, BT, HR). This aligns with clinical
intuition — pain severity and cardiac/respiratory complaint language are well-established triage drivers.
12. Simulated Validation Set (20 Patients)
To stress-test the deployed model beyond the historical test split, 20 simulated patient records were constructed and
passed through the actual trained triage_model.joblib pipeline (not a re-implemented approximation). The outputs
below — predicted class, all 5 class probabilities, confidence, and confidence band — are genuine model output, not
hand-authored numbers.
Automated ED Triage System — Technical Documentation Page 14
Patient
ID(s)
Scenario What it demonstrates
1-13 Baseline 13 diverse presentations spanning all 5 KTAS levels, used as a general
functional check.
14 Ambiguous
presentation
Vague weakness/dizziness, no clear cause. The real model splits its
probability across KTAS 2 (34.5%) and KTAS 3 (59.6%) — genuine,
non-engineered uncertainty.
15 Pediatric 4-year-old with high fever and lethargy; vitals reflect normal pediatric
tachycardia/tachypnea rather than adult reference ranges.
16 Geriatric 88-year-old found down and confused. Model confidence is only 45.8% (Low
band) — a genuine limitation given how few elderly-fall cases exist in
training data, not a scripted failure.
17 Zero-history (first visit) Explicit Prior_Visit_History = 0 flag; chief complaint states this is the
patient's first visit, no baseline records available.
18, 19 Simulated 3× surge Patient 19 repeats patient 5's exact clinical picture, but with
Patients-number-per-hour raised from 7 to 21 (3×). Predicted class stayed
KTAS 3 in both; confidence rose slightly (81.3% → 84.6%).
20 Clinician override Model confidently predicts KTAS 4 (74.1%) for a self-described
"indigestion." Triage nurse overrides to KTAS 2 after noting pallor and
diaphoresis not captured in structured fields; the override reason, reviewer
identity, and timestamp are all logged.
12.1 Key findings from the simulated set
• Uncertainty is genuinely surfaced, not decorative. Five of the twenty simulated patients (IDs 3, 4, 8, 13, 16)
landed in the Low confidence band (<50%) purely from realistic vitals/complaints, without being deliberately
engineered to fail — confirming the confidence mechanism reflects real model uncertainty.
• The surge test reveals a real limitation, stated plainly. Site load (Patients number per hour) has measurable but
small influence on the model (Section 11); tripling it did not change the predicted class for the paired comparison
(patients 5 vs. 19). This means the model does not currently treat a queue-length surge as a safety-relevant signal
in its own right — a finding worth disclosing rather than hiding.
• The override case demonstrates the full audit trail. When the clinician's judgment overrides the model, the
system captures what the model predicted, what the clinician decided, why, who made the decision, and when —
the minimum information needed for later review or model retraining.
Automated ED Triage System — Technical Documentation Page 15
13. System Files & Architecture
File Purpose
KTAS_data_cleaned.xlsx Cleaned, structured source dataset (1,267 patients, 24 columns).
train_triage_model.py End-to-end training script: loads data, builds the preprocessing + model
pipeline, runs cross-validation, evaluates on the held-out test set, and saves
the final artifacts.
triage_model.joblib The trained, ready-to-use scikit-learn Pipeline object plus metadata (feature
lists, model name, test metrics), produced by train_triage_model.py.
metrics_report.txt Plain-text evaluation report generated at training time (the source of all
numbers in Section 10).
confusion_matrix.png /
feature_importance.png
Diagnostic plots generated at training time, reproduced in Sections 10-11.
predict_triage.py Minimal example script showing how to load triage_model.joblib and get a
prediction for one new patient.
app.py Streamlit web application: a form-based interface that loads the trained
model and serves live predictions, without needing to retrain.
Simulated_Triage_Patients_20.xlsx The 20-patient stress-test set described in Section 12, including real model
output and the clinician-override log.
13.1 Data flow
KTAS_data_cleaned.xlsx → train_triage_model.py (cleans column types, builds the preprocessing pipeline,
cross-validates 3 candidate models, evaluates the winner on a held-out test set) → triage_model.joblib (the
deployable artifact) → either predict_triage.py (command-line single-patient prediction) or app.py (interactive web
form) for actual use on new patients.
14. Limitations & Responsible Use
• Not a certified medical device. This is a decision-support prototype. A qualified clinician must make the final
triage call in every case.
• Small, narrow training set. 1,267 patients from what appears to be two hospital sites is not enough data to
guarantee the model generalizes to other hospitals, populations, or countries without local retraining and validation.
• Rare classes are underserved. KTAS 1 and 5 make up only 8% of the data combined; recall on class 5 is just
13.3%. The model needs more examples of the rarest, most extreme cases before its performance on them can be
trusted.
• 13.8% under-triage rate on the historical test set. Roughly 1 in 7 predictions understated urgency relative to the
expert label. Any deployment needs a human triage professional in the loop, not an unsupervised automated
pipeline.
• Surge behavior is limited. As shown in Section 12.1, the model does not meaningfully change its output under a
simulated 3× volume surge, meaning it should not be relied on to detect or respond to system-level capacity strain
on its own.
Automated ED Triage System — Technical Documentation Page 16
• The site-load feature is a static number, not a queueing model. "Patients number per hour" describes historical
conditions in the training data, not real-time wait-time risk, and should not be over-interpreted as a dynamic
surge-detection signal.
In short: this system is well-suited as a second opinion and a documentation aid for human triage staff, with its confidence
scores used to decide when to lean on the model versus when to insist on independent clinical judgment — not as a
replacement for trained triage personnel.
