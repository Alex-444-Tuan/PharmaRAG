# Grounding claim labelling worksheet

## Rubric (read this first)

Each block below is one sentence written by the answer model, the evidence blocks it cites, and
two empty lines for you to fill in.

- Type **S** after `LABEL:` if the sentence is **supported**: every fact in it is stated in, or
  directly follows from, the cited evidence shown in that block.
- Type **H** if it is **hallucinated**: **any** fact in it (a number, a unit, a drug, a
  population, a frequency, a negation, a causal claim) is not stated in or directly entailed by
  the cited evidence. **This holds even if the fact is true in the world.** Medical knowledge
  you hold does not count as evidence.
- Judge against the cited evidence shown, nothing else. A sentence whose support is not in the
  blocks it cites is H, even if some other block of the original answer would have supported it.
- Reasonable paraphrase is fine ("twice daily" for "two times a day"). A changed or extra
  number, a flipped negation, a different population or a stronger claim ("always", "prevents")
  than the evidence makes is not.
- Use the NOTE line for anything you want to remember. It is not read by any script.
- Do not use any other letter. Do not leave a LABEL line empty. Keep the `## Claim` headers and
  the `LABEL:` and `NOTE:` lines as they are: the importer reads them.
- Do not look at the verifier's output before labelling.

---

## Claim g01

QUESTION: What is the usual starting dose of metformin for adults with type 2 diabetes?
SENTENCE: For immediate-release metformin hydrochloride tablets, the recommended starting dose in adults is 500 mg orally twice a day or 850 mg once a day, given with meals.
CITES: c1

EVIDENCE c1 (2 DOSAGE AND ADMINISTRATION > 2.1 Adult Dosage):
> Metformin Hydrochloride Tablets
> • The recommended starting dose of metformin hydrochloride tablets is 500 mg orally twice a day or 850 mg once a day, given with meals.
> • Increase the dose in increments of 500 mg weekly or 850 mg every 2 weeks on the basis of glycemic control and tolerability, up to a maximum dose of 2,550 mg per day, given in divided doses.
> • Doses above 2,000 mg may be better tolerated given 3 times a day with meals.
> Metformin Hydrochloride Extended-Release Tablets
> • Swallow metformin hydrochloride extended-release tablets whole and never crush, cut or chew. • The recommended starting dose of metformin hydrochloride extended-release tablets are 500 mg orally once daily with the evening meal. • Increase the dosage in increments of 500 mg weekly on the basis of glycemic control and tolerability, up to a maximum of 2,000 mg once daily with the evening meal. • If glycemic control is not achieved with metformin hydrochloride extended-release tablets 2,000 mg once daily, consider a trial of metformin hydrochloride extended-release tablets 1,000 mg twice daily.

LABEL:
NOTE:

---

## Claim g02

QUESTION: What are the contraindications for lisinopril?
SENTENCE: Lisinopril (Zestril) is contraindicated in combination with a neprilysin inhibitor such as sacubitril.
CITES: c1

EVIDENCE c1 (4 CONTRAINDICATIONS):
> Zestril is contraindicated in combination with a neprilysin inhibitor (e.g., sacubitril). Do not administer Zestril within 36 hours of switching to or from sacubitril/valsartan, a neprilysin inhibitor [ see Warnings and Precautions (5.2)].
> Zestril is contraindicated in patients with:
> a history of angioedema or hypersensitivity related to previous treatment with an angiotensin converting enzyme inhibitor
> hereditary or idiopathic angioedema
> Do not co-administer aliskiren with ZESTRIL in patients with diabetes [see Drug Interactions (7.4) ].

LABEL:
NOTE:

---

## Claim g03

QUESTION: What are the most common side effects of sertraline?
SENTENCE: In pooled placebo-controlled trials, the most common adverse reactions (≥5% and twice placebo) were nausea, diarrhea/loose stool, tremor, dyspepsia, decreased appetite, hyperhidrosis, ejaculation failure, and decreased libido.
CITES: c5

EVIDENCE c5 (6 ADVERSE REACTIONS > 6.1 Clinical Trials Experience):
> Because clinical trials are conducted under widely varying conditions, adverse reaction rates observed in the clinical trials of a drug cannot be directly compared to rates in the clinical trials of another drug and may not reflect the rates observed in practice.
> The data described below are from randomized, double-blind, placebo-controlled trials of ZOLOFT (mostly 50 mg to 200 mg per day) in 3066 adults diagnosed with MDD, OCD, PD, PTSD, SAD, and PMDD. These 3066 patients exposed to ZOLOFT for 8 to12 weeks represent 568 patient-years of exposure. The mean age was 40 years; 57% were females and 43% were males.
> The most common adverse reactions (≥5% and twice placebo) in all pooled placebo-controlled clinical trials of all ZOLOFT-treated patients with MDD, OCD, PD, PTSD, SAD and PMDD were nausea, diarrhea/loose stool, tremor, dyspepsia, decreased appetite, hyperhidrosis, ejaculation failure, and decreased libido (see Table 3). The following are the most common adverse reactions in trials of ZOLOFT (≥5% and twice placebo) by indication that were not mentioned previously.
> •MDD: somnolence;
> •OCD: insomnia, agitation;
> •PD: constipation, agitation;
> •PTSD: fatigue;
> •PMDD: somnolence, dry mouth, dizziness, fatigue, and abdominal pain;
> •SAD: insomnia, dizziness, fatigue, dry mouth, malaise.

LABEL:
NOTE:

---

## Claim g04

QUESTION: Which medicines raise the bleeding risk when taken together with warfarin?
SENTENCE: The warfarin label lists several drug classes that increase bleeding risk when used with warfarin and says to monitor such patients closely.
CITES: c1

EVIDENCE c1 (7 DRUG INTERACTIONS > 7.3 Drugs that Increase Bleeding Risk):
> Examples of drugs known to increase the risk of bleeding are presented in Table 3. Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin.
> Table 3: Drugs that Can Increase the Risk of Bleeding
> Drug Class: Anticoagulants; Specific Drugs: argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin
> Drug Class: Antiplatelet Agents; Specific Drugs: aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine
> Drug Class: Nonsteroidal Anti-Inflammatory Agents; Specific Drugs: celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac
> Drug Class: Serotonin Reuptake Inhibitors; Specific Drugs: citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, vilazodone

LABEL:
NOTE:

---

## Claim g05

QUESTION: What is the maximum daily dose of tramadol for adults?
SENTENCE: The maximum daily dose of tramadol hydrochloride tablets for adults is 400 mg per day.
CITES: c2, c3

EVIDENCE c2 (2 DOSAGE AND ADMINISTRATION > 2.1 Important Dosage and Administration Instructions):
> Do not use tramadol hydrochloride tablets concomitantly with other tramadol-containing products.
> Do not administer tramadol hydrochloride tablets at a dose exceeding 400 mg per day.
> Tramadol hydrochloride tablets should be prescribed only by healthcare professionals who are knowledgeable about the use of opioids and how to mitigate the associated risks.
> Use the lowest effective dosage for the shortest duration of time consistent with individual patient treatment goals [see Warnings and Precautions (5)]. Because the risk of overdose increases as opioid doses increase, reserve titration to higher doses of tramadol hydrochloride tablets for patients in whom lower doses are insufficiently effective and in whom the expected benefits of using a higher dose opioid clearly outweigh the substantial risks.
> Many acute pain conditions (e.g., the pain that occurs with a number of surgical procedures or acute musculoskeletal injuries) require no more than a few days of an opioid analgesic. Clinical guidelines on opioid prescribing for some acute pain conditions are available.
> There is variability in the opioid analgesic dose and duration needed to adequately manage pain due both to the cause of pain and to individual patient factors. Initiate the dosing regimen for each patient individually, taking into account the patient’s underlying cause and severity of pain, prior analgesic treatment and response, and risk factors for addiction, abuse, and misuse [see Warnings and Precautions (5.1)].

EVIDENCE c3 (2 DOSAGE AND ADMINISTRATION > 2.3 Initial Dosage):
> Initiating Treatment with Tramadol Hydrochloride Tablets
> Initiate treatment at the lowest dose necessary to achieve adequate analgesia. Titrate the dose based upon the individual patient’s response to their initial dose of tramadol hydrochloride tablets.
> For patients not requiring rapid onset of analgesic effect, the tolerability of tramadol hydrochloride tablets can be improved by initiating therapy with the following titration regimen: Start tramadol hydrochloride tablets at 25 mg/day and titrated in 25 mg increments as separate doses every 3 days to reach 100 mg/day (25 mg four times a day). Thereafter the total daily dose may be increased by 50 mg as tolerated every 3 days to reach 200 mg/day (50 mg four times a day). After titration, tramadol hydrochloride tablets 50 mg to 100 mg can be administered as needed for pain relief every 4 to 6 hours not to exceed 400 mg/day.
> For the subset of patients for whom rapid onset of analgesic effect is required and for whom the benefits outweigh the risk of discontinuation due to adverse events associated with higher initial doses, tramadol hydrochloride tablets 50 mg to 100 mg can be administered as needed for pain relief every four to six hours, not to exceed 400 mg per day.
> Conversion from Tramadol Hydrochloride Immediate-Release to Extended-Release Tramadol

LABEL:
NOTE:

---

## Claim g06

QUESTION: Can amlodipine be used during pregnancy?
SENTENCE: The limited post-marketing data on NORVASC use in pregnant women are not sufficient to inform a drug-associated risk for major birth defects and miscarriage.
CITES: c1

EVIDENCE c1 (8 USE IN SPECIFIC POPULATIONS > 8.1 Pregnancy > Risk Summary):
> The limited available data based on post-marketing reports with NORVASC use in pregnant women are not sufficient to inform a drug-associated risk for major birth defects and miscarriage. There are risks to the mother and fetus associated with poorly controlled hypertension in pregnancy [see Clinical Considerations]. In animal reproduction studies, there was no evidence of adverse developmental effects when pregnant rats and rabbits were treated orally with amlodipine maleate during organogenesis at doses approximately 10 and 20-times the maximum recommended human dose (MRHD), respectively. However for rats, litter size was significantly decreased (by about 50%) and the number of intrauterine deaths was significantly increased (about 5-fold). Amlodipine has been shown to prolong both the gestation period and the duration of labor in rats at this dose [see Data].
> The estimated background risk of major birth defects and miscarriage for the indicated population is unknown. All pregnancies have a background risk of birth defect, loss or other adverse outcomes. In the U.S. general population, the estimated background risk of major birth defects and miscarriage in clinically recognized pregnancies is 2%-4% and 15%‑20%, respectively.

LABEL:
NOTE:

---

## Claim g07

QUESTION: What should I know about stopping paroxetine suddenly?
SENTENCE: Stopping paroxetine abruptly can cause discontinuation adverse reactions.
CITES: c1, c2

EVIDENCE c1 (5 WARNINGS AND PRECAUTIONS > 5.7 Discontinuation Syndrome):
> Adverse reactions after discontinuation of serotonergic antidepressants, particularly after abrupt discontinuation, include: nausea, sweating, dysphoric mood, irritability, agitation, dizziness, sensory disturbances (e.g., paresthesia, such as electric shock sensations), tremor, anxiety, confusion, headache, lethargy, emotional lability, insomnia, hypomania, tinnitus, and seizures. A gradual reduction in dosage rather than abrupt cessation is recommended whenever possible [see Dosage and Administration (2.7)].
> During clinical trials of GAD and PTSD, gradual decreases in the daily dose by 10 mg/day at weekly intervals followed by 1 week at 20 mg/day was used before treatment was discontinued. The following adverse reactions were reported at an incidence of 2% or greater for paroxetine and were at least twice that reported for placebo: Abnormal dreams, paresthesia, and dizziness adverse reactions have been reported upon discontinuation of treatment with paroxetine in pediatric patients. The safety and effectiveness of paroxetine in pediatric patients have not been established [see Boxed Warning, Warnings and Precautions (5.1), Use in Specific Populations (8.4)].

EVIDENCE c2 (2 DOSAGE AND ADMINISTRATION > 2.7 Discontinuation of Treatment with Paroxetine Tablets):
> Adverse reactions may occur upon discontinuation of paroxetine tablets [see Warnings and Precautions (5.7)]. Gradually reduce the dosage rather than stopping paroxetine tablets abruptly whenever possible.

LABEL:
NOTE:

---

## Claim g08

QUESTION: How should levothyroxine be taken in relation to food?
SENTENCE: Levothyroxine should be taken as a single daily dose on an empty stomach, one-half to one hour before breakfast.
CITES: c1

EVIDENCE c1 (2 DOSAGE AND ADMINISTRATION > 2.1 Important Administration Instructions):
> Administer Levothyroxine Sodium Tablets as a single daily dose, on an empty stomach, one-half to one hour before breakfast.
> Administer Levothyroxine Sodium Tablets at least 4 hours before or after drugs known to interfere with Levothyroxine Sodium Tablets absorption [see Drug Interactions (7.1)].
> Evaluate the need for dosage adjustments when regularly administering within one hour of certain foods that may affect Levothyroxine Sodium Tablets absorption [see Dosage and Administration (2.2 and 2.3), Drug Interactions (7.9) and Clinical Pharmacology (12.3)].
> Administer Levothyroxine Sodium Tablets to pediatric patients who cannot swallow intact tablets by crushing the tablet, suspending the freshly crushed tablet in a small amount (5 to 10 mL) of water and immediately administering the suspension by spoon or dropper. Ensure the patient ingests the full amount of the suspension. Do not store the suspension. Do not administer in foods that decrease absorption of Levothyroxine Sodium Tablets, such as soybean-based infant formula [see Drug Interactions (7.9)].

LABEL:
NOTE:

---

## Claim g09

QUESTION: What does the fluoxetine label warn about suicidal thoughts in young people?
SENTENCE: The boxed warning states that antidepressants increased the risk of suicidal thoughts and behavior in children, adolescents, and young adults in short-term studies.
CITES: c1

EVIDENCE c1 (WARNING: SUICIDAL THOUGHTS AND BEHAVIORS):
> Antidepressants increased the risk of suicidal thoughts and behavior in children, adolescents, and young adults in short-term studies. These studies did not show an increase in the risk of suicidal thoughts and behavior with antidepressant use in patients over age 24; there was a reduction in risk with antidepressant use in patients aged 65 and older [see Warnings and Precautions (5.1)].
> In patients of all ages who are started on antidepressant therapy, monitor closely for worsening and for emergence of suicidal thoughts and behaviors. Advise families and caregivers of the need for close observation and communication with the prescriber [see Warnings and Precautions (5.1)].
> PROZAC is not approved for use in children less than 7 years of age [see Warnings and Precautions (5.1) and Use in Specific Populations (8.4)].
> When using PROZAC and olanzapine in combination, also refer to Boxed Warning section of the package insert for Symbyax.

LABEL:
NOTE:

---

## Claim g10

QUESTION: Is it safe to take ibuprofen every day if I drink alcohol regularly?
SENTENCE: Use of alcohol is listed among the factors that increase the risk of GI bleeding in patients treated with NSAIDs.
CITES: c3

EVIDENCE c3 (WARNINGS):
> Patients with a prior history of peptic ulcer disease and/or gastrointestinal bleeding who use NSAIDs have a greater than 10-fold increased risk for developing a GI bleed compared to patients treated with neither of these risk factors. Other factors that increase the risk of GI bleeding in patients treated with NSAIDs include concomitant use of oral corticosteroids or anticoagulants, longer duration of NSAID therapy, smoking, use of alcohol, older age, and poor general health status. Most spontaneous reports of fatal GI events are in elderly or debilitated patients and therefore, special care should be taken in treating this population. To minimize the potential risk for an adverse GI event in patients treated with an NSAID, the lowest effective dose should be used for the shortest possible duration. Patients and physicians should remain alert for signs and symptoms of GI ulcerations and bleeding during NSAID therapy and promptly initiate additional evaluation and treatment if a serious GI event is suspected. This should include discontinuation of the NSAID until a serious GI adverse event is ruled out. For high-risk patients, alternate therapies that do not involve NSAIDs should be considered.
> Renal Effects
> Long-term administration of NSAIDs has resulted in renal papillary necrosis and other renal injury. Renal toxicity has also been seen in patients in whom renal prostaglandins have a compensatory role in the maintenance of renal perfusion. In these patients, administration of a NSAID may cause a dose dependent reduction in prostaglandin formation and, secondarily, in renal blood flow, which may precipitate overt renal decompensation.

LABEL:
NOTE:

---

## Claim g11

QUESTION: What is the usual starting dose of metformin for adults with type 2 diabetes?
SENTENCE: For extended-release tablets, the recommended starting dose is 500 mg orally once daily with the evening meal.
CITES: c1

EVIDENCE c1 (2 DOSAGE AND ADMINISTRATION > 2.1 Adult Dosage):
> Metformin Hydrochloride Tablets
> • The recommended starting dose of metformin hydrochloride tablets is 500 mg orally twice a day or 850 mg once a day, given with meals.
> • Increase the dose in increments of 500 mg weekly or 850 mg every 2 weeks on the basis of glycemic control and tolerability, up to a maximum dose of 2,550 mg per day, given in divided doses.
> • Doses above 2,000 mg may be better tolerated given 3 times a day with meals.
> Metformin Hydrochloride Extended-Release Tablets
> • Swallow metformin hydrochloride extended-release tablets whole and never crush, cut or chew. • The recommended starting dose of metformin hydrochloride extended-release tablets are 500 mg orally once daily with the evening meal. • Increase the dosage in increments of 500 mg weekly on the basis of glycemic control and tolerability, up to a maximum of 2,000 mg once daily with the evening meal. • If glycemic control is not achieved with metformin hydrochloride extended-release tablets 2,000 mg once daily, consider a trial of metformin hydrochloride extended-release tablets 1,000 mg twice daily.

LABEL:
NOTE:

---

## Claim g12

QUESTION: What are the contraindications for lisinopril?
SENTENCE: It should not be administered within 36 hours of switching to or from sacubitril/valsartan.
CITES: c1

EVIDENCE c1 (4 CONTRAINDICATIONS):
> Zestril is contraindicated in combination with a neprilysin inhibitor (e.g., sacubitril). Do not administer Zestril within 36 hours of switching to or from sacubitril/valsartan, a neprilysin inhibitor [ see Warnings and Precautions (5.2)].
> Zestril is contraindicated in patients with:
> a history of angioedema or hypersensitivity related to previous treatment with an angiotensin converting enzyme inhibitor
> hereditary or idiopathic angioedema
> Do not co-administer aliskiren with ZESTRIL in patients with diabetes [see Drug Interactions (7.4) ].

LABEL:
NOTE:

---

## Claim g13

QUESTION: What are the most common side effects of sertraline?
SENTENCE: Nausea was the most frequent reaction, occurring in 26% of sertraline-treated patients versus 12% on placebo.
CITES: c3

EVIDENCE c3 (6 ADVERSE REACTIONS > 6.1 Clinical Trials Experience):
> •PTSD: fatigue;
> •PMDD: somnolence, dry mouth, dizziness, fatigue, and abdominal pain;
> •SAD: insomnia, dizziness, fatigue, dry mouth, malaise.
> Table 3: Common Adverse Reactions in Pooled Placebo-Controlled Trials in Adults with MDD, OCD, PD, PTSD, SAD, and PMDDAdverse reactions that occurred greater than 2% in ZOLOFT-treated patients and at least 2% greater in ZOLOFT-treated patients than placebo-treated patients.
> Cardiac disorders
> Palpitations; ZOLOFT (N=3066): 4%; Placebo (N=2293): 2%
> Eye disorders
> Visual impairment; ZOLOFT (N=3066): 4%; Placebo (N=2293): 2%
> Gastrointestinal disorders
> Nausea; ZOLOFT (N=3066): 26%; Placebo (N=2293): 12%
> Diarrhea/Loose stools; ZOLOFT (N=3066): 20%; Placebo (N=2293): 10%
> Dry mouth; ZOLOFT (N=3066): 14%; Placebo (N=2293): 9%
> Dyspepsia; ZOLOFT (N=3066): 8%; Placebo (N=2293): 4%
> Constipation; ZOLOFT (N=3066): 6%; Placebo (N=2293): 4%
> Vomiting; ZOLOFT (N=3066): 4%; Placebo (N=2293): 1%

LABEL:
NOTE:

---

## Claim g14

QUESTION: Which medicines raise the bleeding risk when taken together with warfarin?
SENTENCE: Anticoagulants named are argatroban, dabigatran, bivalirudin, desirudin, heparin, and lepirudin.
CITES: c1

EVIDENCE c1 (7 DRUG INTERACTIONS > 7.3 Drugs that Increase Bleeding Risk):
> Examples of drugs known to increase the risk of bleeding are presented in Table 3. Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin.
> Table 3: Drugs that Can Increase the Risk of Bleeding
> Drug Class: Anticoagulants; Specific Drugs: argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin
> Drug Class: Antiplatelet Agents; Specific Drugs: aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine
> Drug Class: Nonsteroidal Anti-Inflammatory Agents; Specific Drugs: celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac
> Drug Class: Serotonin Reuptake Inhibitors; Specific Drugs: citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, vilazodone

LABEL:
NOTE:

---

## Claim g15

QUESTION: What is the maximum daily dose of tramadol for adults?
SENTENCE: Patients with creatinine clearance below 30 mL/min should have the dosing interval increased to 12 hours, with a maximum daily dose of 200 mg.
CITES: c1

EVIDENCE c1 (2 DOSAGE AND ADMINISTRATION > 2.3 Initial Dosage):
> Conversion from Tramadol Hydrochloride Immediate-Release to Extended-Release Tramadol
> The relative bioavailability of immediate-release tramadol hydrochloride tablets compared to extended-release tramadol is unknown, so conversion to extended-release formulations may lead to increased risk of excessive sedation and respiratory depression.
> Dosage Modification in Patients with Hepatic Impairment
> The recommended dose for adult patients with severe hepatic impairment is 50 mg every 12 hours.
> Dosage Modification in Patients with Renal Impairment
> In all patients with creatinine clearance less than 30 mL/min, it is recommended that the dosing interval of tramadol hydrochloride tablets be increased to 12 hours, with a maximum daily dose of 200 mg. Since only 7% of an administered dose is removed by hemodialysis, dialysis patients can receive their regular dose on the day of dialysis.
> Dosage Modification in Geriatric Patients
> Do not exceed a total dose of 300 mg/day in patients over 75 years old.

LABEL:
NOTE:

---

## Claim g16

QUESTION: Can amlodipine be used during pregnancy?
SENTENCE: Poorly controlled hypertension in pregnancy carries risks to both the mother and the fetus.
CITES: c1

EVIDENCE c1 (8 USE IN SPECIFIC POPULATIONS > 8.1 Pregnancy > Risk Summary):
> The limited available data based on post-marketing reports with NORVASC use in pregnant women are not sufficient to inform a drug-associated risk for major birth defects and miscarriage. There are risks to the mother and fetus associated with poorly controlled hypertension in pregnancy [see Clinical Considerations]. In animal reproduction studies, there was no evidence of adverse developmental effects when pregnant rats and rabbits were treated orally with amlodipine maleate during organogenesis at doses approximately 10 and 20-times the maximum recommended human dose (MRHD), respectively. However for rats, litter size was significantly decreased (by about 50%) and the number of intrauterine deaths was significantly increased (about 5-fold). Amlodipine has been shown to prolong both the gestation period and the duration of labor in rats at this dose [see Data].
> The estimated background risk of major birth defects and miscarriage for the indicated population is unknown. All pregnancies have a background risk of birth defect, loss or other adverse outcomes. In the U.S. general population, the estimated background risk of major birth defects and miscarriage in clinically recognized pregnancies is 2%-4% and 15%‑20%, respectively.

LABEL:
NOTE:

---

## Claim g17

QUESTION: What should I know about stopping paroxetine suddenly?
SENTENCE: Reported reactions after abrupt discontinuation of serotonergic antidepressants include nausea, sweating, dysphoric mood, irritability, agitation, dizziness, tremor, anxiety, confusion, headache, lethargy, emotional lability, insomnia, hypomania, tinnitus, and seizures.
CITES: c1

EVIDENCE c1 (5 WARNINGS AND PRECAUTIONS > 5.7 Discontinuation Syndrome):
> Adverse reactions after discontinuation of serotonergic antidepressants, particularly after abrupt discontinuation, include: nausea, sweating, dysphoric mood, irritability, agitation, dizziness, sensory disturbances (e.g., paresthesia, such as electric shock sensations), tremor, anxiety, confusion, headache, lethargy, emotional lability, insomnia, hypomania, tinnitus, and seizures. A gradual reduction in dosage rather than abrupt cessation is recommended whenever possible [see Dosage and Administration (2.7)].
> During clinical trials of GAD and PTSD, gradual decreases in the daily dose by 10 mg/day at weekly intervals followed by 1 week at 20 mg/day was used before treatment was discontinued. The following adverse reactions were reported at an incidence of 2% or greater for paroxetine and were at least twice that reported for placebo: Abnormal dreams, paresthesia, and dizziness adverse reactions have been reported upon discontinuation of treatment with paroxetine in pediatric patients. The safety and effectiveness of paroxetine in pediatric patients have not been established [see Boxed Warning, Warnings and Precautions (5.1), Use in Specific Populations (8.4)].

LABEL:
NOTE:

---

## Claim g18

QUESTION: How should levothyroxine be taken in relation to food?
SENTENCE: The need for dosage adjustments should be evaluated when it is regularly taken within one hour of certain foods that may affect its absorption.
CITES: c1

EVIDENCE c1 (2 DOSAGE AND ADMINISTRATION > 2.1 Important Administration Instructions):
> Administer Levothyroxine Sodium Tablets as a single daily dose, on an empty stomach, one-half to one hour before breakfast.
> Administer Levothyroxine Sodium Tablets at least 4 hours before or after drugs known to interfere with Levothyroxine Sodium Tablets absorption [see Drug Interactions (7.1)].
> Evaluate the need for dosage adjustments when regularly administering within one hour of certain foods that may affect Levothyroxine Sodium Tablets absorption [see Dosage and Administration (2.2 and 2.3), Drug Interactions (7.9) and Clinical Pharmacology (12.3)].
> Administer Levothyroxine Sodium Tablets to pediatric patients who cannot swallow intact tablets by crushing the tablet, suspending the freshly crushed tablet in a small amount (5 to 10 mL) of water and immediately administering the suspension by spoon or dropper. Ensure the patient ingests the full amount of the suspension. Do not store the suspension. Do not administer in foods that decrease absorption of Levothyroxine Sodium Tablets, such as soybean-based infant formula [see Drug Interactions (7.9)].

LABEL:
NOTE:

---

## Claim g19

QUESTION: What does the fluoxetine label warn about suicidal thoughts in young people?
SENTENCE: Those studies showed no increased risk in patients over age 24, and a reduction in risk in patients aged 65 and older.
CITES: c1

EVIDENCE c1 (WARNING: SUICIDAL THOUGHTS AND BEHAVIORS):
> Antidepressants increased the risk of suicidal thoughts and behavior in children, adolescents, and young adults in short-term studies. These studies did not show an increase in the risk of suicidal thoughts and behavior with antidepressant use in patients over age 24; there was a reduction in risk with antidepressant use in patients aged 65 and older [see Warnings and Precautions (5.1)].
> In patients of all ages who are started on antidepressant therapy, monitor closely for worsening and for emergence of suicidal thoughts and behaviors. Advise families and caregivers of the need for close observation and communication with the prescriber [see Warnings and Precautions (5.1)].
> PROZAC is not approved for use in children less than 7 years of age [see Warnings and Precautions (5.1) and Use in Specific Populations (8.4)].
> When using PROZAC and olanzapine in combination, also refer to Boxed Warning section of the package insert for Symbyax.

LABEL:
NOTE:

---

## Claim g20

QUESTION: Is it safe to take ibuprofen every day if I drink alcohol regularly?
SENTENCE: NSAIDs, including ibuprofen, can cause serious GI events such as bleeding, ulceration, and perforation, which can be fatal and can occur at any time, with or without warning symptoms.
CITES: c1, c2

EVIDENCE c1 (WARNINGS):
> Avoid the use of ibuprofen tablets in patients with severe heart failure unless the benefits are expected to outweigh the risk of worsening heart failure. If ibuprofen tablets is used in patients with severe heart failure, monitor patients for signs of worsening heart failure.
> Gastrointestinal Effects - Risk of Ulceration, Bleeding, and Perforation
> NSAIDs, including ibuprofen tablets, can cause serious gastrointestinal (GI) adverse events including inflammation, bleeding, ulceration, and perforation of the stomach, small intestine, or large intestine, which can be fatal. These serious adverse events can occur at any time, with or without warning symptoms, in patients treated with NSAIDs. Only one in five patients, who develop a serious upper GI adverse event on NSAID therapy, is symptomatic. Upper GI ulcers, gross bleeding, or perforation caused by NSAIDs occur in approximately 1% of patients treated for 3-6 months, and in about 2 to 4% of patients treated for one year. These trends continue with longer duration of use, increasing the likelihood of developing a serious GI event at some time during the course of therapy. However, even short-term therapy is not without risk. NSAIDs should be prescribed with extreme caution in those with a prior history of ulcer disease or gastrointestinal bleeding. Patients with a prior history of peptic ulcer disease and/or gastrointestinal bleeding who use NSAIDs have a greater than 10-fold increased risk for developing a GI bleed compared to patients treated with neither of these risk factors.

EVIDENCE c2 (BOXED WARNING):
> Cardiovascular Thrombotic Events
> Nonsteroidal anti-inflammatory drugs (NSAIDs) cause an increased risk of serious cardiovascular thrombotic events, including myocardial infarction and stroke, which can be fatal. This risk may occur early in treatment and may increase with duration of use. (See WARNINGS and PRECAUTIONS).
> Ibuprofen tablets are contraindicated in the setting of coronary artery bypass graft (CABG) surgery (See CONTRAINDICATIONS and WARNINGS).
> Gastrointestinal Risk
> NSAIDS cause an increased risk of serious gastrointestinal adverse events including bleeding, ulceration, and perforation of the stomach or intestines, which can be fatal. These events can occur at any time during use and without warning symptoms. Elderly patients are at greater risk for serious gastrointestinal events. (See WARNINGS).

LABEL:
NOTE:

---

## Claim g21

QUESTION: What are the contraindications for lisinopril?
SENTENCE: It is contraindicated in patients with a history of angioedema or hypersensitivity related to previous treatment with an angiotensin converting enzyme inhibitor.
CITES: c1

EVIDENCE c1 (4 CONTRAINDICATIONS):
> Zestril is contraindicated in combination with a neprilysin inhibitor (e.g., sacubitril). Do not administer Zestril within 36 hours of switching to or from sacubitril/valsartan, a neprilysin inhibitor [ see Warnings and Precautions (5.2)].
> Zestril is contraindicated in patients with:
> a history of angioedema or hypersensitivity related to previous treatment with an angiotensin converting enzyme inhibitor
> hereditary or idiopathic angioedema
> Do not co-administer aliskiren with ZESTRIL in patients with diabetes [see Drug Interactions (7.4) ].

LABEL:
NOTE:

---

## Claim g22

QUESTION: What are the most common side effects of sertraline?
SENTENCE: Diarrhea/loose stools occurred in 20% versus 10% on placebo, and dry mouth in 14% versus 9%.
CITES: c3

EVIDENCE c3 (6 ADVERSE REACTIONS > 6.1 Clinical Trials Experience):
> •PTSD: fatigue;
> •PMDD: somnolence, dry mouth, dizziness, fatigue, and abdominal pain;
> •SAD: insomnia, dizziness, fatigue, dry mouth, malaise.
> Table 3: Common Adverse Reactions in Pooled Placebo-Controlled Trials in Adults with MDD, OCD, PD, PTSD, SAD, and PMDDAdverse reactions that occurred greater than 2% in ZOLOFT-treated patients and at least 2% greater in ZOLOFT-treated patients than placebo-treated patients.
> Cardiac disorders
> Palpitations; ZOLOFT (N=3066): 4%; Placebo (N=2293): 2%
> Eye disorders
> Visual impairment; ZOLOFT (N=3066): 4%; Placebo (N=2293): 2%
> Gastrointestinal disorders
> Nausea; ZOLOFT (N=3066): 26%; Placebo (N=2293): 12%
> Diarrhea/Loose stools; ZOLOFT (N=3066): 20%; Placebo (N=2293): 10%
> Dry mouth; ZOLOFT (N=3066): 14%; Placebo (N=2293): 9%
> Dyspepsia; ZOLOFT (N=3066): 8%; Placebo (N=2293): 4%
> Constipation; ZOLOFT (N=3066): 6%; Placebo (N=2293): 4%
> Vomiting; ZOLOFT (N=3066): 4%; Placebo (N=2293): 1%

LABEL:
NOTE:

---

## Claim g23

QUESTION: Which medicines raise the bleeding risk when taken together with warfarin?
SENTENCE: Antiplatelet agents named are aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, and ticlopidine.
CITES: c1

EVIDENCE c1 (7 DRUG INTERACTIONS > 7.3 Drugs that Increase Bleeding Risk):
> Examples of drugs known to increase the risk of bleeding are presented in Table 3. Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin.
> Table 3: Drugs that Can Increase the Risk of Bleeding
> Drug Class: Anticoagulants; Specific Drugs: argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin
> Drug Class: Antiplatelet Agents; Specific Drugs: aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine
> Drug Class: Nonsteroidal Anti-Inflammatory Agents; Specific Drugs: celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac
> Drug Class: Serotonin Reuptake Inhibitors; Specific Drugs: citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, vilazodone

LABEL:
NOTE:

---

## Claim g24

QUESTION: What is the maximum daily dose of tramadol for adults?
SENTENCE: In patients over 75 years old, the total dose should not exceed 300 mg/day.
CITES: c1

EVIDENCE c1 (2 DOSAGE AND ADMINISTRATION > 2.3 Initial Dosage):
> Conversion from Tramadol Hydrochloride Immediate-Release to Extended-Release Tramadol
> The relative bioavailability of immediate-release tramadol hydrochloride tablets compared to extended-release tramadol is unknown, so conversion to extended-release formulations may lead to increased risk of excessive sedation and respiratory depression.
> Dosage Modification in Patients with Hepatic Impairment
> The recommended dose for adult patients with severe hepatic impairment is 50 mg every 12 hours.
> Dosage Modification in Patients with Renal Impairment
> In all patients with creatinine clearance less than 30 mL/min, it is recommended that the dosing interval of tramadol hydrochloride tablets be increased to 12 hours, with a maximum daily dose of 200 mg. Since only 7% of an administered dose is removed by hemodialysis, dialysis patients can receive their regular dose on the day of dialysis.
> Dosage Modification in Geriatric Patients
> Do not exceed a total dose of 300 mg/day in patients over 75 years old.

LABEL:
NOTE:

---

## Claim g25

QUESTION: Can amlodipine be used during pregnancy?
SENTENCE: Hypertension in pregnancy raises the maternal risk for pre-eclampsia, gestational diabetes, premature delivery, and delivery complications, and raises the fetal risk for intrauterine growth restriction and intrauterine death.
CITES: c2

EVIDENCE c2 (8 USE IN SPECIFIC POPULATIONS > 8.1 Pregnancy > Clinical Considerations > Disease-associated maternal and/or embryo/fetal risk):
> Hypertension in pregnancy increases the maternal risk for pre-eclampsia, gestational diabetes, premature delivery, and delivery complications (e.g., need for cesarean section and post-partum hemorrhage). Hypertension increases the fetal risk for intrauterine growth restriction and intrauterine death. Pregnant women with hypertension should be carefully monitored and managed accordingly.

LABEL:
NOTE:

---

## Claim g26

QUESTION: What should I know about stopping paroxetine suddenly?
SENTENCE: Sensory disturbances such as paresthesia, including electric shock sensations, are also among the reactions listed.
CITES: c1

EVIDENCE c1 (5 WARNINGS AND PRECAUTIONS > 5.7 Discontinuation Syndrome):
> Adverse reactions after discontinuation of serotonergic antidepressants, particularly after abrupt discontinuation, include: nausea, sweating, dysphoric mood, irritability, agitation, dizziness, sensory disturbances (e.g., paresthesia, such as electric shock sensations), tremor, anxiety, confusion, headache, lethargy, emotional lability, insomnia, hypomania, tinnitus, and seizures. A gradual reduction in dosage rather than abrupt cessation is recommended whenever possible [see Dosage and Administration (2.7)].
> During clinical trials of GAD and PTSD, gradual decreases in the daily dose by 10 mg/day at weekly intervals followed by 1 week at 20 mg/day was used before treatment was discontinued. The following adverse reactions were reported at an incidence of 2% or greater for paroxetine and were at least twice that reported for placebo: Abnormal dreams, paresthesia, and dizziness adverse reactions have been reported upon discontinuation of treatment with paroxetine in pediatric patients. The safety and effectiveness of paroxetine in pediatric patients have not been established [see Boxed Warning, Warnings and Precautions (5.1), Use in Specific Populations (8.4)].

LABEL:
NOTE:

---

## Claim g27

QUESTION: How should levothyroxine be taken in relation to food?
SENTENCE: For pediatric patients who cannot swallow tablets, the crushed tablet suspension should not be given in foods that decrease absorption, such as soybean-based infant formula.
CITES: c1

EVIDENCE c1 (2 DOSAGE AND ADMINISTRATION > 2.1 Important Administration Instructions):
> Administer Levothyroxine Sodium Tablets as a single daily dose, on an empty stomach, one-half to one hour before breakfast.
> Administer Levothyroxine Sodium Tablets at least 4 hours before or after drugs known to interfere with Levothyroxine Sodium Tablets absorption [see Drug Interactions (7.1)].
> Evaluate the need for dosage adjustments when regularly administering within one hour of certain foods that may affect Levothyroxine Sodium Tablets absorption [see Dosage and Administration (2.2 and 2.3), Drug Interactions (7.9) and Clinical Pharmacology (12.3)].
> Administer Levothyroxine Sodium Tablets to pediatric patients who cannot swallow intact tablets by crushing the tablet, suspending the freshly crushed tablet in a small amount (5 to 10 mL) of water and immediately administering the suspension by spoon or dropper. Ensure the patient ingests the full amount of the suspension. Do not store the suspension. Do not administer in foods that decrease absorption of Levothyroxine Sodium Tablets, such as soybean-based infant formula [see Drug Interactions (7.9)].

LABEL:
NOTE:

---

## Claim g28

QUESTION: What does the fluoxetine label warn about suicidal thoughts in young people?
SENTENCE: Pooled analyses of short-term placebo-controlled trials found that antidepressants increase suicidal thinking and behavior in children, adolescents, and young adults (ages 18-24) with MDD and other psychiatric disorders.
CITES: c2

EVIDENCE c2 (5 WARNINGS AND PRECAUTIONS > 5.1 Suicidal Thoughts and Behaviors in Children, Adolescents, and Young Adults):
> Patients with Major Depressive Disorder (MDD), both adult and pediatric, may experience worsening of their depression and/or the emergence of suicidal ideation and behavior (suicidality) or unusual changes in behavior, whether or not they are taking antidepressant medications, and this risk may persist until significant remission occurs. Suicide is a known risk of depression and certain other psychiatric disorders, and these disorders themselves are the strongest predictors of suicide. There has been a long-standing concern, however, that antidepressants may have a role in inducing worsening of depression and the emergence of suicidality in certain patients during the early phases of treatment. Pooled analyses of short-term placebo-controlled trials of antidepressant drugs (SSRIs and others) showed that these drugs increase the risk of suicidal thinking and behavior (suicidality) in children, adolescents, and young adults (ages 18-24) with Major Depressive Disorder (MDD) and other psychiatric disorders. Short-term studies did not show an increase in the risk of suicidality with antidepressants compared to placebo in adults beyond age 24; there was a reduction with antidepressants compared to placebo in adults aged 65 and older.
> The pooled analyses of placebo-controlled trials in children and adolescents with MDD, Obsessive Compulsive Disorder (OCD), or other psychiatric disorders included a total of 24 short-term trials of 9 antidepressant drugs in over 4400 patients.

LABEL:
NOTE:

---

## Claim g29

QUESTION: Is it safe to take ibuprofen every day if I drink alcohol regularly?
SENTENCE: Risk rises with longer duration of therapy, with about 1% of patients having serious upper GI events over 3-6 months and about 2 to 4% over one year.
CITES: c1

EVIDENCE c1 (WARNINGS):
> Avoid the use of ibuprofen tablets in patients with severe heart failure unless the benefits are expected to outweigh the risk of worsening heart failure. If ibuprofen tablets is used in patients with severe heart failure, monitor patients for signs of worsening heart failure.
> Gastrointestinal Effects - Risk of Ulceration, Bleeding, and Perforation
> NSAIDs, including ibuprofen tablets, can cause serious gastrointestinal (GI) adverse events including inflammation, bleeding, ulceration, and perforation of the stomach, small intestine, or large intestine, which can be fatal. These serious adverse events can occur at any time, with or without warning symptoms, in patients treated with NSAIDs. Only one in five patients, who develop a serious upper GI adverse event on NSAID therapy, is symptomatic. Upper GI ulcers, gross bleeding, or perforation caused by NSAIDs occur in approximately 1% of patients treated for 3-6 months, and in about 2 to 4% of patients treated for one year. These trends continue with longer duration of use, increasing the likelihood of developing a serious GI event at some time during the course of therapy. However, even short-term therapy is not without risk. NSAIDs should be prescribed with extreme caution in those with a prior history of ulcer disease or gastrointestinal bleeding. Patients with a prior history of peptic ulcer disease and/or gastrointestinal bleeding who use NSAIDs have a greater than 10-fold increased risk for developing a GI bleed compared to patients treated with neither of these risk factors.

LABEL:
NOTE:

---

## Claim g30

QUESTION: What are the contraindications for lisinopril?
SENTENCE: It is contraindicated in patients with hereditary or idiopathic angioedema.
CITES: c1

EVIDENCE c1 (4 CONTRAINDICATIONS):
> Zestril is contraindicated in combination with a neprilysin inhibitor (e.g., sacubitril). Do not administer Zestril within 36 hours of switching to or from sacubitril/valsartan, a neprilysin inhibitor [ see Warnings and Precautions (5.2)].
> Zestril is contraindicated in patients with:
> a history of angioedema or hypersensitivity related to previous treatment with an angiotensin converting enzyme inhibitor
> hereditary or idiopathic angioedema
> Do not co-administer aliskiren with ZESTRIL in patients with diabetes [see Drug Interactions (7.4) ].

LABEL:
NOTE:

---

## Claim g31

QUESTION: What are the most common side effects of sertraline?
SENTENCE: Dyspepsia occurred in 8% versus 4% on placebo, and constipation in 6% versus 4%.
CITES: c3

EVIDENCE c3 (6 ADVERSE REACTIONS > 6.1 Clinical Trials Experience):
> •PTSD: fatigue;
> •PMDD: somnolence, dry mouth, dizziness, fatigue, and abdominal pain;
> •SAD: insomnia, dizziness, fatigue, dry mouth, malaise.
> Table 3: Common Adverse Reactions in Pooled Placebo-Controlled Trials in Adults with MDD, OCD, PD, PTSD, SAD, and PMDDAdverse reactions that occurred greater than 2% in ZOLOFT-treated patients and at least 2% greater in ZOLOFT-treated patients than placebo-treated patients.
> Cardiac disorders
> Palpitations; ZOLOFT (N=3066): 4%; Placebo (N=2293): 2%
> Eye disorders
> Visual impairment; ZOLOFT (N=3066): 4%; Placebo (N=2293): 2%
> Gastrointestinal disorders
> Nausea; ZOLOFT (N=3066): 26%; Placebo (N=2293): 12%
> Diarrhea/Loose stools; ZOLOFT (N=3066): 20%; Placebo (N=2293): 10%
> Dry mouth; ZOLOFT (N=3066): 14%; Placebo (N=2293): 9%
> Dyspepsia; ZOLOFT (N=3066): 8%; Placebo (N=2293): 4%
> Constipation; ZOLOFT (N=3066): 6%; Placebo (N=2293): 4%
> Vomiting; ZOLOFT (N=3066): 4%; Placebo (N=2293): 1%

LABEL:
NOTE:

---

## Claim g32

QUESTION: Which medicines raise the bleeding risk when taken together with warfarin?
SENTENCE: NSAIDs named are celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, and sulindac.
CITES: c1

EVIDENCE c1 (7 DRUG INTERACTIONS > 7.3 Drugs that Increase Bleeding Risk):
> Examples of drugs known to increase the risk of bleeding are presented in Table 3. Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin.
> Table 3: Drugs that Can Increase the Risk of Bleeding
> Drug Class: Anticoagulants; Specific Drugs: argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin
> Drug Class: Antiplatelet Agents; Specific Drugs: aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine
> Drug Class: Nonsteroidal Anti-Inflammatory Agents; Specific Drugs: celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac
> Drug Class: Serotonin Reuptake Inhibitors; Specific Drugs: citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, vilazodone

LABEL:
NOTE:

---

## Claim g33

QUESTION: Can amlodipine be used during pregnancy?
SENTENCE: Pregnant women with hypertension should be carefully monitored and managed accordingly.
CITES: c2

EVIDENCE c2 (8 USE IN SPECIFIC POPULATIONS > 8.1 Pregnancy > Clinical Considerations > Disease-associated maternal and/or embryo/fetal risk):
> Hypertension in pregnancy increases the maternal risk for pre-eclampsia, gestational diabetes, premature delivery, and delivery complications (e.g., need for cesarean section and post-partum hemorrhage). Hypertension increases the fetal risk for intrauterine growth restriction and intrauterine death. Pregnant women with hypertension should be carefully monitored and managed accordingly.

LABEL:
NOTE:

---

## Claim g34

QUESTION: What should I know about stopping paroxetine suddenly?
SENTENCE: The label recommends gradually reducing the dosage rather than stopping abruptly whenever possible.
CITES: c1, c2

EVIDENCE c1 (5 WARNINGS AND PRECAUTIONS > 5.7 Discontinuation Syndrome):
> Adverse reactions after discontinuation of serotonergic antidepressants, particularly after abrupt discontinuation, include: nausea, sweating, dysphoric mood, irritability, agitation, dizziness, sensory disturbances (e.g., paresthesia, such as electric shock sensations), tremor, anxiety, confusion, headache, lethargy, emotional lability, insomnia, hypomania, tinnitus, and seizures. A gradual reduction in dosage rather than abrupt cessation is recommended whenever possible [see Dosage and Administration (2.7)].
> During clinical trials of GAD and PTSD, gradual decreases in the daily dose by 10 mg/day at weekly intervals followed by 1 week at 20 mg/day was used before treatment was discontinued. The following adverse reactions were reported at an incidence of 2% or greater for paroxetine and were at least twice that reported for placebo: Abnormal dreams, paresthesia, and dizziness adverse reactions have been reported upon discontinuation of treatment with paroxetine in pediatric patients. The safety and effectiveness of paroxetine in pediatric patients have not been established [see Boxed Warning, Warnings and Precautions (5.1), Use in Specific Populations (8.4)].

EVIDENCE c2 (2 DOSAGE AND ADMINISTRATION > 2.7 Discontinuation of Treatment with Paroxetine Tablets):
> Adverse reactions may occur upon discontinuation of paroxetine tablets [see Warnings and Precautions (5.7)]. Gradually reduce the dosage rather than stopping paroxetine tablets abruptly whenever possible.

LABEL:
NOTE:

---

## Claim g35

QUESTION: How should levothyroxine be taken in relation to food?
SENTENCE: Co-administered food is one of the factors that dosing depends on.
CITES: c2

EVIDENCE c2 (2 DOSAGE AND ADMINISTRATION > 2.2 Important Considerations for Dosing):
> The dosage of Levothyroxine Sodium Tablets for hypothyroidism or pituitary TSH suppression depends on a variety of factors including: the patient's age, body weight, cardiovascular status, concomitant medical conditions (including pregnancy), concomitant medications, co-administered food and the specific nature of the condition being treated [see Dosage and Administration (2.3), Warnings and Precautions (5), and Drug Interactions (7)]. Dosing must be individualized to account for these factors and dosage adjustments made based on periodic assessment of the patient's clinical response and laboratory parameters [see Dosage and Administration (2.4)].
> For adult patients with primary hypothyroidism, titrate until the patient is clinically euthyroid and the serum TSH returns to normal [see Dosage and Administration (2.3)].
> For secondary or tertiary hypothyroidism, serum TSH is not a reliable measure of Levothyroxine Sodium Tablets dosage adequacy and should not be used to monitor therapy. Use the serum free-T4 level to titrate Levothyroxine Sodium Tablets dosing until the patient is clinically euthyroid and the serum free-T4 level is restored to the upper half of the normal range [see Dosage and Administration (2.3)].
> The peak therapeutic effect of a given dose of Levothyroxine Sodium Tablets may not be attained for 4 to 6 weeks.

LABEL:
NOTE:

---

## Claim g36

QUESTION: What does the fluoxetine label warn about suicidal thoughts in young people?
SENTENCE: Compared with placebo, the drug-placebo difference was 14 additional cases of suicidality per 1000 patients treated under age 18, and 5 additional cases per 1000 at ages 18-24.
CITES: c4

EVIDENCE c4 (5 WARNINGS AND PRECAUTIONS > 5.1 Suicidal Thoughts and Behaviors in Children, Adolescents, and Young Adults):
> The pooled analyses of placebo-controlled trials in adults with MDD or other psychiatric disorders included a total of 295 short-term trials (median duration of 2 months) of 11 antidepressant drugs in over 77,000 patients. There was considerable variation in risk of suicidality among drugs, but a tendency toward an increase in the younger patients for almost all drugs studied. There were differences in absolute risk of suicidality across the different indications, with the highest incidence in MDD. The risk differences (drug versus placebo), however, were relatively stable within age strata and across indications. These risk differences (drug-placebo difference in the number of cases of suicidality per 1000 patients treated) are provided in Table 2.
> Table 2: Suicidality per 1000 Patients Treated
> Increases Compared to Placebo
> Age Range: <18; Drug-Placebo Difference in Number of Cases of Suicidality per 1000 Patients Treated: 14 additional cases
> Age Range: 18-24; Drug-Placebo Difference in Number of Cases of Suicidality per 1000 Patients Treated: 5 additional cases
> Decreases Compared to Placebo
> Age Range: 25-64; Drug-Placebo Difference in Number of Cases of Suicidality per 1000 Patients Treated: 1 fewer case
> Age Range: ≥65; Drug-Placebo Difference in Number of Cases of Suicidality per 1000 Patients Treated: 6 fewer cases
> No suicides occurred in any of the pediatric trials. There were suicides in the adult trials, but the number was not sufficient to reach any conclusion about drug effect on suicide.
> It is unknown whether the suicidality risk extends to longer-term use, i.e., beyond several months.

LABEL:
NOTE:

---

## Claim g37

QUESTION: Is it safe to take ibuprofen every day if I drink alcohol regularly?
SENTENCE: Other factors that increase GI bleeding risk include longer duration of NSAID therapy, smoking, older age, and concomitant corticosteroids or anticoagulants.
CITES: c3

EVIDENCE c3 (WARNINGS):
> Patients with a prior history of peptic ulcer disease and/or gastrointestinal bleeding who use NSAIDs have a greater than 10-fold increased risk for developing a GI bleed compared to patients treated with neither of these risk factors. Other factors that increase the risk of GI bleeding in patients treated with NSAIDs include concomitant use of oral corticosteroids or anticoagulants, longer duration of NSAID therapy, smoking, use of alcohol, older age, and poor general health status. Most spontaneous reports of fatal GI events are in elderly or debilitated patients and therefore, special care should be taken in treating this population. To minimize the potential risk for an adverse GI event in patients treated with an NSAID, the lowest effective dose should be used for the shortest possible duration. Patients and physicians should remain alert for signs and symptoms of GI ulcerations and bleeding during NSAID therapy and promptly initiate additional evaluation and treatment if a serious GI event is suspected. This should include discontinuation of the NSAID until a serious GI adverse event is ruled out. For high-risk patients, alternate therapies that do not involve NSAIDs should be considered.
> Renal Effects
> Long-term administration of NSAIDs has resulted in renal papillary necrosis and other renal injury. Renal toxicity has also been seen in patients in whom renal prostaglandins have a compensatory role in the maintenance of renal perfusion. In these patients, administration of a NSAID may cause a dose dependent reduction in prostaglandin formation and, secondarily, in renal blood flow, which may precipitate overt renal decompensation.

LABEL:
NOTE:

---

## Claim g38

QUESTION: What are the contraindications for lisinopril?
SENTENCE: Aliskiren should not be co-administered with Zestril in patients with diabetes.
CITES: c1

EVIDENCE c1 (4 CONTRAINDICATIONS):
> Zestril is contraindicated in combination with a neprilysin inhibitor (e.g., sacubitril). Do not administer Zestril within 36 hours of switching to or from sacubitril/valsartan, a neprilysin inhibitor [ see Warnings and Precautions (5.2)].
> Zestril is contraindicated in patients with:
> a history of angioedema or hypersensitivity related to previous treatment with an angiotensin converting enzyme inhibitor
> hereditary or idiopathic angioedema
> Do not co-administer aliskiren with ZESTRIL in patients with diabetes [see Drug Interactions (7.4) ].

LABEL:
NOTE:

---

## Claim g39

QUESTION: What are the most common side effects of sertraline?
SENTENCE: Hyperhidrosis occurred in 7% of sertraline-treated patients versus 3% on placebo.
CITES: c7

EVIDENCE c7 (6 ADVERSE REACTIONS > 6.1 Clinical Trials Experience):
> Erectile dysfunction; ZOLOFT (N=3066): 4%; Placebo (N=2293): 1%
> Ejaculation disorder; ZOLOFT (N=3066): 3%; Placebo (N=2293): 0%
> Male sexual dysfunctiom; ZOLOFT (N=3066): 2%; Placebo (N=2293): 0%
> Skin and subcutaneous tissue disorders
> Hyperhidrosis; ZOLOFT (N=3066): 7%; Placebo (N=2293): 3%
> Adverse Reactions Leading to Discontinuation in Placebo-Controlled Clinical Trials
> In all placebo-controlled studies in patients with MDD, OCD, PD, PTSD, SAD and PMDD, 368 (12%) of the 3066 patients who received ZOLOFT discontinued treatment due to an adverse reaction, compared with 93 (4%) of the 2293 placebo-treated patients. In placebo-controlled studies, the following were the common adverse reactions leading to discontinuation in ZOLOFT-treated patients:
> •MDD, OCD, PD, PTSD, SAD and PMDD: nausea (3%), diarrhea (2%), agitation (2%), and insomnia (2%).
> •MDD (>2% and twice placebo): decreased appetite, dizziness, fatigue, headache, somnolence, tremor, and vomiting.
> •OCD: somnolence.
> •PD: nervousness and somnolence.
> Male and Female Sexual Dysfunction

LABEL:
NOTE:

---

## Claim g40

QUESTION: Which medicines raise the bleeding risk when taken together with warfarin?
SENTENCE: Serotonin reuptake inhibitors named are citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, and vilazodone.
CITES: c1

EVIDENCE c1 (7 DRUG INTERACTIONS > 7.3 Drugs that Increase Bleeding Risk):
> Examples of drugs known to increase the risk of bleeding are presented in Table 3. Because bleeding risk is increased when these drugs are used concomitantly with warfarin, closely monitor patients receiving any such drug with warfarin.
> Table 3: Drugs that Can Increase the Risk of Bleeding
> Drug Class: Anticoagulants; Specific Drugs: argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin
> Drug Class: Antiplatelet Agents; Specific Drugs: aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine
> Drug Class: Nonsteroidal Anti-Inflammatory Agents; Specific Drugs: celecoxib, diclofenac, diflunisal, fenoprofen, ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, piroxicam, sulindac
> Drug Class: Serotonin Reuptake Inhibitors; Specific Drugs: citalopram, desvenlafaxine, duloxetine, escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, vilazodone

LABEL:
NOTE:

---
