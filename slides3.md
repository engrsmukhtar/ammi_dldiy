
class: center, middle

# AMMI Deep Learning DIY:
# Day 1
Alexandre Sablayrolles, Pierre Stock

---
name:objectives
# Deep Learning DIY: Objectives

Given an idea or a project, be able to implement it in a robust way

---
template:objectives
- Ex: "Given a set of images, can I find the best classifier ?"
<div style="width: 100%;">
  <img src="figs/cifar.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:objectives
- Ex: "With a conversation dataset, how can I create a chatbot ? How can I build a translation model ?"
<div style="width: 100%;">
  <img src="figs/xlm.jpeg" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:objectives
- Ex: "I don't know anything about speech recognition. Can I read a couple of papers and get started on a model ?"
<div style="float:left; width: 49%;">
  <img src="figs/speech_rec.jpg" style="vertical-align:center;width:80%;height: 100%;"/>
</div>
<div style="float:right; width: 49%;">
  <img src="figs/deep_speech.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:objectives
- Write code from scratch

--
 - Write a training loop, a dataloader, etc.

--
 - Already know what to do (generic machine learning code) when the project starts

---
template:objectives
- Write code from scratch
- Debug the project, from code to machine learning assumptions

--
  - did I use the right parameters (learning rate, optimizer, etc.) ?

--
  - am I sure that my evaluation is correct ?

--
  - are the training and test set different ?

--
  - my model does not work on this data, does it work on other data ?

---
template:objectives
- Write code from scratch
- Debug the project, from code to machine learning assumptions
- Top-down approach to understanding
  - there are a lot of moving parts, but we nail them down progressively

---
# Teachers

Pierre Stock

---
# Teachers

Timothée Lacroix
---
# Teachers

Alexandre Sablayrolles
---
# Schedule

First week
- Monday
- Tuesday
- Wednesday
- Thursday
- Friday

Second week
- Monday
- Tuesday
- Wednesday
- Thursday
- Friday
---

# Outline

1. The machine learning stack
2. Brief recap: gaussian random variables
3. Logistic regression and linear classification
4. How to debug a machine learning code?

---
class: outline

# Outline

<ol>
<li class="outline_current">The machine learning stack</li>
<li>Brief recap: gaussian random variables</li>
<li>Logistic regression and linear classification</li>
<li>How to debug a machine learning code?</li>
</ol>

---

name:machine_learning_stack
# The various levels of machine learning...

<div style="float:left; width: 49%">
  <img src="figs/machine_learning_stack.svg" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:machine_learning_stack

<div style="float:right; width: 49%;">
  <img src="figs/headline.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:machine_learning_stack

<div style="float:right; width: 49%;">
  <img src="figs/table_results.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:machine_learning_stack

<div style="float:right; width: 49%;">
  <img src="figs/model.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:machine_learning_stack

<div style="float:right; width: 49%;">
  <img src="figs/pytorch.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:machine_learning_stack

<div style="float:right; width: 49%;">
  <img src="figs/cuda.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>
---
template:machine_learning_stack

<div style="float:right; width: 49%;">
  <img src="figs/geforce.jpg" style="vertical-align:center;width:80%;height: 100%;"/>
</div>
---
name:machine_learning_bugs
# ... and their corresponding bugs


<div style="float:left; width: 49%">
  <img src="figs/machine_learning_stack.svg" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
template:machine_learning_bugs
<div style="float:right; width: 49%;">
<br />
example:
</div>

---
template:machine_learning_bugs
<div style="float:right; width: 49%;">
<br />
<br />
<br />
example:
</div>

---
class: outline

# Outline

<ol>
<li>The machine learning stack</li>
<li class="outline_current">Brief recap: gaussian random variables</li>
<li>Logistic regression and linear classification</li>
<li>How to debug a machine learning code?</li>
</ol>

---
class: outline

# Outline

<ol>
<li>The machine learning stack</li>
<li>Brief recap: gaussian random variables</li>
<li class="outline_current">Logistic regression and linear classification</li>
<li>How to debug a machine learning code?</li>
</ol>

---

# Logistic regression

- We want to separate two gaussians
- Generating process: for a class \\(y \in \\{0, 1\\}\\), we generate \\(x\\):
$$ P(x|y=0) = C \exp(-\\| x - \mu_0 \\|^2) $$
$$ P(x|y=1) = C \exp(-\\| x - \mu_1 \\|^2) $$


---
# Logistic regression
- In machine learning, we observe \\(x\\) and want to infer \\(y\\)
$$ P(y=1|x) = ? $$

--
- We can apply Bayes'rule:
$$ P(y=1|x) = \frac{P(y=1, x)}{P(x)} $$
$$ P(y=1|x) = \frac{\textcolor{blue}{P(x|y=1)P(y=1)}}{\textcolor{blue}{P(x|y=1)P(y=1)} + \textcolor{green}{P(x|y=0)P(y=0)}} $$

---
# Logistic regression
- Logistic function \\( \sigma(x) = 1 / (1 + \exp(-x)) \\)

--
- We have:
$$ \frac{\textcolor{blue}{a}}{\textcolor{blue}{a} + \textcolor{green}{b}} = \sigma \left( -\log \left( \frac{\textcolor{green}{b}}{\textcolor{blue}{a}}\right) \right) $$

--
- Hence:
$$ P(y=1|x) = \sigma \left( -\log \left( \frac{\textcolor{green}{P(x|y=0)P(y=0)}}{\textcolor{blue}{P(x|y=1)P(y=1)}}\right) \right) $$

---
# Logistic regression
- We assume balanced classes: \\(\textcolor{green}{P(y=0)} = \textcolor{blue}{P(y=1)}\\)

--
- Let's work out the term inside \\( \sigma \\):
$$ -\log \left( \frac{\textcolor{green}{P(x|y=0)P(y=0)}}{\textcolor{blue}{P(x|y=1)P(y=1)}} \right) = -\log \left( \frac{\textcolor{green}{C \exp(-\frac{1}{2}\\|x-\mu_0\\|^2)}}{\textcolor{blue}{C \exp(-\frac{1}{2}\\|x-\mu_1\\|^2)}} \right) $$

--
$$ \qquad\qquad\qquad\qquad\qquad\qquad\;\, = \textcolor{green}{\frac{1}{2}\\|x-\mu_0\\|^2} - \textcolor{blue}{\frac{1}{2}\\|x-\mu_1\\|^2} $$

--
$$ \qquad\qquad\qquad\qquad\qquad\qquad\;\, = \langle x, \mu_1 - \mu_0 \rangle + \frac{ \\| \mu_1 \\|^2 - \\| \mu_0 \\|^2 }{2}  $$

--
- Our loss function is
$$ -\log \left( P(y=1|x) \right) = -\log \left( \sigma \left(\langle x, \mu_1 - \mu_0 \rangle + \frac{ \\| \mu_1 \\|^2 - \\| \mu_0 \\|^2 }{2} \right) \right)$$


---
# Logistic regression

- Our loss function writes as:
$$ -\log \left( P(y=1|x) \right) = f \left( \langle x, w \rangle + b  \right)$$
where \\(w = \mu_1 - \mu_0\\), \\(\quad b =\frac{ \\| \mu_1 \\|^2 - \\| \mu_0 \\|^2 }{2} \\), and:
$$ f(t) = -\log(\sigma(t)) $$
$$ \qquad \qquad \qquad \;\; = - \log \left( \frac{1}{1 + \exp(-x)} \right) $$
$$ \qquad \qquad \;\; = \log(1 + \exp(-x)) $$



---
# Logistic regression

<div class="frame">
  <span class="helper"></span>
  <img src="figs/obiwan.jpg" style="vertical-align:middle"/>
</div>


---
# Why the maths

- Why we do the maths
  - __Understand__: why do we do logistic regression ?
  - __Debug__: Gaussians are a simple case to test our code
  - __Visualize__: put some sense back into equations
- In summary, binary logistic regression minimizes \\( f \left( \langle x, w \rangle + b  \right) \\), with \\(f(t) = \log(1 + \exp(-x))\\)

---
# What does the loss look like ?

<div style="width: 98%">
  <img src="figs/losses.png" style="vertical-align:center;width:80%;height: 100%;"/>
</div>

---
# Gradients

- How do we learn the model ?
