For SIH 2025 Problem Statement 50, we’re building a smart traffic system that cuts urban congestion using reinforcement learning, SUMO simulations, OpenCV’s sharp-eyed computer vision, and IoT devices working in sync.📖 Problem Statement

Traffic congestion arises due to unbalanced signal timing, fluctuating traffic density, and the limited adaptability of traditional controllers. Traffic jams build up when signal lights aren’t timed evenly, traffic flow keeps changing, and old-style controllers can’t adjust on the fly—like cars bunching at a green light that stays red too long for the cross street.⚙️ Implementation
Data Collection: Roadside IoT sensors (cameras, inductive loops, GPS-enabled vehicles) capture real-time traffic data.
We use adaptive, data-driven methods that blend computer vision with IoT, letting the system make smart traffic calls on the spot—like rerouting cars when a camera spots sudden congestion.⚙️ Implementation  
Roadside IoT sensors—cameras catching passing headlights, inductive loops under the asphalt, and GPS-equipped vehicles—gather live traffic data in real time. With computer vision, OpenCV crunches live video feeds to spot and follow vehicles, gauge how long the line stretches, and sort them into types—like a bus rumbling past or a small red hatchback. IoT data fusion pulls in streams—air quality readings, shifting weather patterns, even the quick blink of a V2I signal—and blends them into one detailed state vector. RL Controller: Trained in SUMO through TraCI, the RL agent fine-tunes green splits, adjusts phase lengths, and shifts offsets on the fly—like catching the green just as the crosswalk light clicks over.✨ Key Features
Adaptive Signal Control: RL-driven timing that continuously learns from live data.
The reward function aims to cut down wait times, curb emissions, and push throughput to its peak—like keeping traffic lights green just long enough for a smooth flow. By shuffling domain variables, it builds resilience that holds up whether the sky’s clear or rain taps against the window.✨ Key Features  
Adaptive Signal Control: RL-powered timing that keeps adjusting as it learns from real-time traffic, right down to the sound of a car’s tires on wet asphalt. IoT-enabled data fusion pulls together readings from GPS, on-board sensors, and nearby connected cars, like merging a map’s pinpoint with the hum of an engine.🌍 Impact

By uniting RL, SUMO, OpenCV, and IoT, our system evolves beyond static traffic control to deliver a sustainable, intelligent, and responsive solution for modern smart cities.

👥 Team

Developed by Team CROSSFAIR for SIH 2025.
