# DM i AI 2025
<h3>Welcome to the annual <a href="https://dmiai.dk/">Danish National Championship in AI</a> hosted by <a href="https://ambolt.io/">Ambolt AI</a>, <a href="https://ddsa.dk/">Danish Data Science Academy</a> and <a href="https://www.aicentre.dk/">Pioneer Centre for Artificial Intelligence</a></h3> <br>
In this repository, you will find all the information needed to participate in the event. Please read this in full before proceeding to the use cases, and please make sure to read the full description of every use case. You will be granted points for every use case that you provide a submission for and a total score will be calculated based on the individual submissions. <br> <br>

<h2>Use cases</h2>
Below you can find the three use cases for the DM i AI 2025 event. <br>
Within each use case, you find a description together with a template that can be used to setup an API endpoint. <br> 
The API endpoint will be used for submission and is required. The requirements for the API endpoints are specified in the respective use cases. <br> <br>
<a href="https://github.com/amboltio/DM-i-AI-2025/tree/main/race-car">- Race car</a> <br>
<a href="https://github.com/amboltio/DM-i-AI-2025/tree/main/emergency-healthcare-rag">- Emergency Healthcare RAG</a> <br>
<a href="https://github.com/amboltio/DM-i-AI-2025/tree/main/tumor-segmentation">- Tumor segmentation</a> <br> <br>

Clone this GitHub repository to download templates for all three use cases.
```
git clone https://github.com/amboltio/DM-i-AI-2025.git
```
The use cases have been built on top of the <a href="https://fastapi.tiangolo.com/">FastAPI</a> framework, and can be used to specify endpoints in every use case.

<h2>Discord Server</h2>
Come hang out and talk to other competitors of the event on our Discord server. Discuss the use cases with each other, or get in touch with the organizers to solve issues or questions that may arise during the competition. <a href="https://discord.gg/VK9tZkxt99">Join here!</a> <br>

You can expect an answer from the organizers within a couple of hours on work days within 08:00 to 16:00. In the weekend, you can ask your fellow contestants for help.


<h2>Getting started</h2>
You can check the individual template and find the requirements for the different API endpoints. These have to be exactly the same for the evaluation service to work. Inside ```<use-case>/models/dtos.py``` you can find information on the request and response DTOs, describing the input and output requirements for your API.

<h2>Submission</h2>
When you are ready for submission, head over to the <a href="https://cases.dmiai.dk">Submission Form</a> and submit your solution for a use case by providing the host address for your API and the API key we have provided to you. Make sure that you have tested and validated your connection to the API before you submit! 

**You can only submit a single attempt per use case to the evaluation server**, but as many times as you like to the validation server. We therefore highly recommend that you validate your solution before submitting. You can do this on the submission form by using the `QUEUE VALIDATION ATTEMPT` button. When you queue a validation attempt, your score will show up on the scoreboard, so you can see how you compare to the other teams.

Note: When you validate your solution on the submission form, it will be evaluated on a validation dataset. But when you submit your solution and get the final score for that use case, your solution will be evaluated on an **evaluation dataset which is different from the validation dataset**. This means that the score you obtained when validating your solution may be different from the score you get when evaluating. Therefore, we encourage you not to overfit to the validation set!

<h3>Ranked score and total score </h3>
The scoreboard will display a score for each use case and a "total score".
The individual score reflects the placement your best model has achieved relative to the other participants' models.

We use a ranking system inspired by the <a href="https://en.wikipedia.org/wiki/List_of_Formula_One_World_Championship_points_scoring_systems">Formula 1 points scoring system</a>, which means that the best entry in each use case is awarded 25 points, the second-best entry is awarded 18 points, and each additional entry gets a decreasing award. 

The full point system is as follows:

1) 25 points
2) 18 points
3) 15 points
4) 12 points
5) 10 points
6) 8 points
7) 6 points
8) 4 points
9) 2 points
10) 1 point
11) $>$ 1 point

Rank 11-end are awarded points in the range from 1 to 0. 


The total score is simply the sum of your individual scores.<br>

This format also means that you can lose points / be overtaken by other teams during the week if they submit a model that is better than yours. 

<h3>Deadline for submission</h3>
The deadline for submission is: August 8, 2025 at 14.00

<h3>Final evaluation</h3>
Upon completion of the contest, the top 5 highest-ranking teams will be asked to submit their training code and the trained models for validation no later than August 8th at 16:00. The submissions will be validated by our Scientific Jury who will get back to everyone within top 5 to let them know their placement. 

<h2>How to get a server for deployment?</h2>
When you are doing the submission, we are expecting you to host the server at which the REST API can be deployed. 

As a DM i AI contestant, you have access to compute resources from UCloud. Each team is allocated resources on the following hardware:

- NVIDIA L4 GPU (24 GB VRAM)
- NVIDIA L40 GPU (48 GB VRAM)

Please reach out to the competition admins on discord if your team has not received an invitation link to UCloud within the first day of the competition.


<h3>Azure for Students</h3>
You can sign up to <a href="https://azure.microsoft.com/da-dk/free/students/">Azure for Students</a>, where you will get free credits that you can use to create a virtual machine. We expect you all to be able to do this, since the competition is only for students. Alternatively, you can also deploy your submission locally (This requires a public IP). <br> 
The following contains the necessary links for creating a virtual machine: <br> <br>

* <a href="https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal">Creating a linux virtual machine</a> <br>
* <a href="https://docs.microsoft.com/en-us/azure/virtual-machines/linux/use-remote-desktop">Install and configure xrdp to use Remote Desktop</a> <br>
* <a href="https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal#create-an-inbound-security-rule">Create an inbound security Rule</a> (This ensures that the API endpoints can be accessed when submitting)<br> <br>

<b>Please make sure to get a server up and running early in the competition, and make sure to get connection to the evaluation service as quickly as possible, so if you have any server related issues, we can catch them early and not close to deadline!</b>


<h2>Frequently Asked Questions</h2>

**Q: Can I use a pretrained model I found on the internet?**

**A:** Yes you are allowed to use pretrained models. If you can find a pretrained model fitting your purpose, you would save a lot of time, just like you would do if you were solving a problem for a company.

**Q: Should we gather our own data?**

**A:** This depends on the individual use case. If you believe you can create a better model with more data, you should go gather the data yourself. We are only supplying a limited amount of data, as we want you to get creative in your approach to each use case.  

**Please note, that we do not provide servers for training!** You are expected to train your models and solutions using your own hardware, Google Colab, etc.

**Q: Are we allowed to use OpenAI / Google Cloud / AWS / Azure APIs?**

**A:** No, you are not allowed to use cloud APIs during inference. You ARE allowed to use as many cloud APIs that you want to build your models. When we call the /predict endpoints of your services, however, the models should be able to run on their own without additional cloud API calls.