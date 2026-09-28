# SUBTEXT — Cross-Context Pragmatic Impact Analysis
**Challenge 03:** Context-Aware Sarcasm & Irony Detection
**Model Architecture:** DeBERTa-v3-base across all configurations

This report demonstrates how conversational context changes prediction and model confidence
for identical target responses across the three fine-tuned input strategies:
- **Experiment A:** Response Only
- **Experiment B:** Parent Context + Response
- **Experiment C:** Full Dialogue Context + Response

---
## Example Validation Sample (ID: SARC_TR_02511)
- **Parent Context:** *"The puns and satirical comments of redditors on my serious posts"*
- **Full Dialogue:** *"What part of Reddit culture annoys you? [TURN] The puns and satirical comments of redditors on my serious posts"*
- **Target Response:** *"The fact that the serious tag needs to exist bothers me."*
- **Actual Label:** `NOT_SARCASM`

| Strategy | Context Provided | Predicted Label | Sarcasm Probability | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **Response Only** | *(None - Response Isolated)* | `NOT_SARCASM` | `0.1307` | **86.9%** |
| **Parent + Response** | *The puns and satirical comments of redditors on my serious p...* | `NOT_SARCASM` | `0.1857` | **81.4%** |
| **Full Dialogue + Response** | *What part of Reddit culture annoys you? [TURN] The puns and ...* | `NOT_SARCASM` | `0.2557` | **74.4%** |

---
## Example Validation Sample (ID: SARC_TR_01109)
- **Parent Context:** *"Thy should put that warning at the start of porn sites where you just check if you're 18 or not, that should satisfy some people"*
- **Full Dialogue:** *"Can everyone please stop saying this is illegal? [TURN] Thy should put that warning at the start of porn sites where you just check if you're 18 or not, that should satisfy some people"*
- **Target Response:** *"Haha so true but the kids betting is concerning"*
- **Actual Label:** `NOT_SARCASM`

| Strategy | Context Provided | Predicted Label | Sarcasm Probability | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **Response Only** | *(None - Response Isolated)* | `SARCASM` | `0.6424` | **64.2%** |
| **Parent + Response** | *Thy should put that warning at the start of porn sites where...* | `SARCASM` | `0.6563` | **65.6%** |
| **Full Dialogue + Response** | *Can everyone please stop saying this is illegal? [TURN] Thy ...* | `SARCASM` | `0.8017` | **80.2%** |

---
## Example Validation Sample (ID: SARC_TR_00235)
- **Parent Context:** *"Also the car with the world's fastest speeding ticket."*
- **Full Dialogue:** *"Coming through [TURN] That's a Koenigsegg which is also the only car The Stig ever crashed. [TURN] Also the car with the world's fastest speeding ticket."*
- **Target Response:** *"But it had the Top Gear wing, how could it crash?"*
- **Actual Label:** `SARCASM`

| Strategy | Context Provided | Predicted Label | Sarcasm Probability | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **Response Only** | *(None - Response Isolated)* | `SARCASM` | `0.9357` | **93.6%** |
| **Parent + Response** | *Also the car with the world's fastest speeding ticket.* | `SARCASM` | `0.9440` | **94.4%** |
| **Full Dialogue + Response** | *Coming through [TURN] That's a Koenigsegg which is also the ...* | `SARCASM` | `0.9128` | **91.3%** |

---
## Example Validation Sample (ID: SARC_TR_02294)
- **Parent Context:** *"I'm sure the community is going to be outraged that an innocent young man was killed by evil pigs."*
- **Full Dialogue:** *"Officer Fatally Shoots Armed 14-year-old in Bronx [TURN] I'm sure the community is going to be outraged that an innocent young man was killed by evil pigs."*
- **Target Response:** *"Cant wait for all of the 'walks against police brutality'"*
- **Actual Label:** `NOT_SARCASM`

| Strategy | Context Provided | Predicted Label | Sarcasm Probability | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **Response Only** | *(None - Response Isolated)* | `SARCASM` | `0.9271` | **92.7%** |
| **Parent + Response** | *I'm sure the community is going to be outraged that an innoc...* | `SARCASM` | `0.8114` | **81.1%** |
| **Full Dialogue + Response** | *Officer Fatally Shoots Armed 14-year-old in Bronx [TURN] I'm...* | `NOT_SARCASM` | `0.3108` | **68.9%** |

---
## Example Validation Sample (ID: SARC_TR_00794)
- **Parent Context:** *"Does this mean Wolf Link doesn't do anything?"*
- **Full Dialogue:** *"Check out what the new BoTW series amiibo can do in the upcoming game [TURN] Does this mean Wolf Link doesn't do anything?"*
- **Target Response:** *"Wolf Link gives you Wolf Link in the game to run around with you."*
- **Actual Label:** `NOT_SARCASM`

| Strategy | Context Provided | Predicted Label | Sarcasm Probability | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **Response Only** | *(None - Response Isolated)* | `NOT_SARCASM` | `0.1651` | **83.5%** |
| **Parent + Response** | *Does this mean Wolf Link doesn't do anything?* | `NOT_SARCASM` | `0.1055` | **89.5%** |
| **Full Dialogue + Response** | *Check out what the new BoTW series amiibo can do in the upco...* | `NOT_SARCASM` | `0.2383` | **76.2%** |

---