# Taxonomy sample for human review (blind)

Seeded random sample of 30 incidents coded as data-readiness
failures by `experiments/code_incidents.py` (seed=42, git=6d7d7c3).

**Instructions.** For each incident, without looking at the machine key,
assign ONE mode and a pre-training detectability. This gives an
independent second coding for inter-rater reliability
(`experiments/code_agreement.py`).

Modes: `structural_schema`, `completeness`, `distributional`, `label`,
`provenance_documentation`, `cross_source`, or `other` (not a
data-readiness failure — this also validates the filter's precision).

Detectability (can a pre-training validation gate catch it?):
`yes` / `partial` / `no`.

Fill the two blanks per incident:

---

## Incident 13
**Title:** High-Toxicity Assessed on Text Involving Women and Minority Groups

Google's Perspective API, which assigns a toxicity score to online text, seems to award higher toxicity scores to content involving non-white, male, Christian, heterosexual phrases.

- Your mode: `completeness`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 22
**Title:** Waze Navigates Motorists into Wildfires

Waze, a Google-owned directions app, led California drivers into the 2017 Skirball wildfires as they tried to evacuate the area.

- Your mode: `structural_schema`
- Your detectability: `no`
- (optional) note: 

---

## Incident 36
**Title:** Picture of Woman on Side of Bus Shamed for Jaywalking

Facial recognition system in China mistakes celebrity's face on moving billboard for jaywalker

- Your mode: `completeness`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 37
**Title:** Amazon’s Experimental Hiring Tool Allegedly Displayed Gender Bias in Candidate Rankings

Between 2014 and 2017, Amazon reportedly developed an AI-powered recruiting tool to score job applicants, trained on a decade of resumes purportedly drawn largely from men. Media reports say the system learned to favor male candidates, penalizing terms like "women's" and graduates from certain all-women's colleges. Efforts to remove these biases reportedly did not guarantee fairness, and the project was ultimately abandoned. Amazon reportedly states recruiters never solely relied on the tool.

- Your mode: `label`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 74
**Title:** Detroit Police Wrongfully Arrested Black Man Due To Faulty FRT

A Black man was wrongfully detained by the Detroit Police Department as a result of a false facial recognition (FRT) result.

- Your mode: `cross_source`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 82
**Title:** #LekkiMassacre: Why Facebook labelled content from October 20 incident ‘false’

Facebook incorrectly labels content relating to an incident between #EndSARS protestors and the Nigerian army as misinformation.

- Your mode: `label`
- Your detectability: `no`
- (optional) note: 

---

## Incident 103
**Title:** Twitter’s Image Cropping Tool Allegedly Showed Gender and Racial Bias

Twitter's photo cropping algorithm was revealed by researchers to favor white and women faces in photos containing multiple faces, prompting the company to stop its use on mobile platform.

- Your mode: `structural_schema`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 146
**Title:** Research Prototype AI, Delphi, Reportedly Gave Racially Biased Answers on Ethics

A publicly accessible research model that was trained via Reddit threads showed racially biased advice on moral dilemmas, allegedly demonstrating limitations of language-based models trained on moral judgments.

- Your mode: `completeness`
- Your detectability: `no`
- (optional) note: 

---

## Incident 149
**Title:** Zillow Shut Down Zillow Offers Division Allegedly Due to Predictive Pricing Tool's Insufficient Accuracy

Zillow's AI-powered predictive pricing tool Zestimate was allegedly not able to accurately forecast housing prices three to six months in advance due to rapid market changes, prompting division shutdown and layoff of a few thousand employees.

- Your mode: `other`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 150
**Title:** Swedish Contraceptive App, Natural Cycles, Allegedly Failed to Correctly Map Menstrual Cycle

Some women using the contraceptive app, Natural Cycles, reported unwanted pregnancies, revealing its algorithm's difficulties in mapping menstrual cycles.

- Your mode: `cross_source`
- Your detectability: `no`
- (optional) note: 

---

## Incident 154
**Title:** Justice Department’s Recidivism Risk Algorithm PATTERN Allegedly Caused Persistent Disparities Along Racial Lines

Department of Justice’s inmate-recidivism risk assessment tool was reported to have produced racially uneven results, misclassifying risk levels for inmates of color.

- Your mode: `structural_schema`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 166
**Title:** Networking Platform Giggle Employs AI to Determine Users’ Gender, Allegedly Excluding Transgender Women

A social networking platform, Giggle, allegedly collected, shared to third-parties, and used sensitive information and biometric data to verify whether a person is a woman via facial recognition, which critics claimed to be discriminatory against women of color and harmful towards trans women.

- Your mode: `other`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 188
**Title:** Argentinian City Government Deployed Teenage-Pregnancy Predictive Algorithm Using Invasive Demographic Data

In 2018, during the abortion-decriminalization debate in Argentina, the Salta city government deployed a teenage-pregnancy predictive algorithm built by Microsoft that allegedly lacked a defined purpose, explicitly considered sensitive information such as disability and whether their home had access to hot water.

- Your mode: `structural_schema`
- Your detectability: `no`
- (optional) note: 

---

## Incident 199
**Title:** Ever AI Reportedly Deceived Customers about FRT Use in App

Ever AI, now Paravision AI, allegedly failed to inform customers about the development and use of facial recognition that facilitates the sale of customers’ data to various businesses, a business model that critics said was an egregious violation of privacy.

- Your mode: `distributional`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 226
**Title:** Waze Allegedly Clogged Streets and Directed Drivers to Make Unsafe Traffic Decisions

For years, Waze has, in an attempt to cut travel times, allegedly caused more traffic and guided drivers to make unsafe and often un-permitted traffic decisions, which was described by a Los Angeles city council member as a threat to public safety.

- Your mode: `distributional`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 234
**Title:** Waze Allegedly Frequently Routed Drivers through the Town of Los Gatos, Blocking Its Single Wildfire Escape Route

Waze app was blamed by Los Gatos town residents for contributing to high wildfire hazard risk via allegedly routing weekend beach-going drivers through their neighborhoods, effectively choking off their single escape route in the event of a medical emergency or wildfire.

- Your mode: `cross_source`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 244
**Title:** Colorado Police’s Automated License Plate Reader (ALPR) Matched a Family’s Minivan’s Plate to That of a Stolen Vehicle Allegedly, Resulting in Detainment at Gunpoint

An automated plate reader reportedly matched a license plate information, but of a family’s minivan and an alleged motorcycle in Montana that was reportedly stolen earlier in the year, resulting in them and their children being held at gunpoint and detained in handcuffs by multiple Aurora police officers.

- Your mode: `structural_schema`
- Your detectability: `no`
- (optional) note: 

---

## Incident 254
**Title:** Google’s Face Grouping Allegedly Collected and Analyzed Users’ Facial Structure without Consent, Violated BIPA

A class-action lawsuit alleged Google failing to provide notice, obtain informed written consent, or publish data retention policies about the collection, storage, and analysis of its face-grouping feature in Google Photos, which violated Illinois Biometric Information Privacy Act (BIPA).

- Your mode: `distributional`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 265
**Title:** Black Uber Eats Driver Allegedly Subjected to Excessive Photo Checks and Dismissed via FRT Results

A lawsuit by a former Uber Eats delivery driver alleged the company to have wrongfully dismissed him due to frequent false mismatches of his verification selfies, and discriminated against him via excessive verification checks.

- Your mode: `cross_source`
- Your detectability: `no`
- (optional) note: 

---

## Incident 274
**Title:** Virginia Courts’ Algorithmic Recidivism Risk Assessment Failed to Lower Incarceration Rates

Virginia courts’ use of algorithmic predictions of future offending risks were found by researchers failing to reduce incarceration rates, showed racial and age disparities in risk scores and its application, and neither exacerbated or ameliorated historical racial differences in sentencing.

- Your mode: `structural_schema`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 416
**Title:** Facebook's Job Ad Algorithm Allegedly Biased against Older and Female Workers

Facebook's algorithm was alleged in a complaint by Real Women in Trucking to have selectively shown job advertisements disproportionately against older and female workers in favor of younger men for blue-collar positions.

- Your mode: `provenance_documentation`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 452
**Title:** ChatGPT-Written Bug Reports Deemed "Nonsense" by White Hat Platform, Prompted Bans

ChatGPT-generated responses submitted to smart contract bug bounty platform Immunefi reportedly lacked details to help diagnose technical issues, which reportedly wasted the platform's time, prompting bans to submitters.

- Your mode: `distributional`
- Your detectability: `no`
- (optional) note: 

---

## Incident 582
**Title:** Racial Bias in Lung Function Diagnostic Algorithm Leads to Underdiagnosis in Black Men

A study published in JAMA Network Open reveals that racial bias built into a commonly used medical diagnostic algorithm for lung function may be leading to underdiagnoses of breathing problems in Black men. The study suggests that as many as 40% more Black male patients might have been accurately diagnosed if the software were not racially biased. The software algorithm adjusts diagnostic thresholds based on race, affecting medical treatments and interventions.

- Your mode: `structural_schema`
- Your detectability: `no`
- (optional) note: 

---

## Incident 584
**Title:** Illinois Residents File Class Action Lawsuit Against Facial Recognition Technology Companies for Allegedly Violating BIPA

A class action lawsuit was filed against several facial recognition technology companies for allegedly violating the Illinois Biometric Information Privacy Act (BIPA). The defendants are accused of offering a facial recognition search engine called Pimeyes, which collects images from databases across the internet and scans them into their database seemingly without consent. This action is claimed to invade the privacy of millions of Americans. The lawsuit argues that Pimeyes lacks publicly available policies regarding data storage and deletion, in contravention of BIPA's requirements for informed written consent before collecting biometric data.

- Your mode: `provenance_documentation`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 636
**Title:** AI Romance Apps Reportedly Compromise User Privacy for Data Harvesting

AI-powered romantic chatbots, marketed for enhancing mental health, are found to exploit user privacy by harvesting sensitive personal information for data sharing and targeted ads, with inadequate security measures and consent protocols, according to research by the Mozilla Foundation.

- Your mode: `provenance_documentation`
- Your detectability: `no`
- (optional) note: 

---

## Incident 687
**Title:** Deepfake Porn Sites Use Breeze Liu's Image Without Consent

Porn sites are alleged to have used AI-generated images of Breeze Liu without her consent, leading to severe emotional distress. Liu discovered a video of herself on Pornhub, which was then deepfaked and spread across over 800 links. Despite efforts to remove the content, many sites refused to comply, perpetuating the violation and exploitation of her image.

- Your mode: `distributional`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 781
**Title:** Clearview AI Reportedly Faces $33.7 Million Fine for Violating GDPR with Biometric Data Harvesting

Clearview AI was reportedly fined $33.7 million by the Dutch data protection authority for allegedly creating an illegal facial recognition database by scraping billions of images from the Internet without consent. The company allegedly used AI to convert these images into biometric data and sold the service to law enforcement. This act was reportedly in violation of privacy laws and the GDPR.

- Your mode: `completeness`
- Your detectability: `yes`
- (optional) note: 

---

## Incident 882
**Title:** AI-Generated Reading Summaries on Fable App Reportedly Wrote Biased and Offensive Commentary

Fable, a book-focused social app, used OpenAI’s API to generate AI-powered year-end reading summaries in December 2024. These summaries allegedly produced biased and offensive remarks about race, gender, and sexual orientation. Fable is reported to have apologized in a social media post on January 1, 2025.

- Your mode: `completeness`
- Your detectability: `partial`
- (optional) note: 

---

## Incident 1000
**Title:** Sora Video Generator Has Reportedly Been Creating Biased Human Representations Across Race, Gender, and Disability

A WIRED investigation found that OpenAI’s video generation model, Sora, exhibits representational bias across race, gender, body type, and disability. In tests using 250 prompts, Sora was more likely to depict CEOs and professors as men, flight attendants and childcare workers as women, and showed limited or stereotypical portrayals of disabled individuals and people with larger bodies.

- Your mode: `distributional`
- Your detectability: `no`
- (optional) note: 

---

## Incident 1043
**Title:** Reddit Moderators Report Unauthorized AI Study Involving Fabricated Identities by Purported University of Zurich Researchers

Researchers purportedly affiliated with the University of Zurich reportedly deployed undisclosed AI-generated comments on Reddit's r/ChangeMyView to study persuasion by allegedly fabricating identities such as sexual assault survivors and racial minorities. The experiment reportedly involved unauthorized demographic profiling, emotional manipulation, and violations of subreddit and platform rules. The researchers allegedly deviated from their approved protocol without renewed ethics oversight.

- Your mode: `label`
- Your detectability: `yes`
- (optional) note: 

---
