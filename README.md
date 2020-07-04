# Decision Making with Reinforcement Learning For a Ride Hailing Company
### Business Objective
Develop an RL model to assist Cab drivers. This is to maximise profits for the industry while also help retain and encourage more cab drivers. 
The goal of the project is to maximise cab driver profits and help with the decision making process. 

### Model Development
1. Develop the environment:
  -  The cab drivers are using electric cars that run for 30 days non-stop, after which they need to recharge
  - The cab services operates only in 5 locations in the city. And at one point of time, the driver can recieve a maximum of 15 requests. The number of requests per location is calculated basis Poisson distribution.
  - The travel time depends on the day of the week and time of the day (the details for same were provided before hand in the TimeMatrix document)
  - The time is only calculated in hours. 
  - The profit/loss of a ride (reward function) is calculated: (revenue earned from pickup point 𝑝 to drop point 𝑞) - (Cost of
battery used in moving from pickup point 𝑝 to drop point 𝑞) - (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝).
2. Using Architecture 2 (where the input is the state, and output is the probability of the respective actions)
3. Track for Convergence
 
