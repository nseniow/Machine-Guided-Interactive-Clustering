# Interactive Constrained Clustering
 Clustering algorithms are very popular methods. However, these algorithms usually need to be tuned by the end users to correspond to some preconceptions. This process is iterative and time consuming. The goal of this project is to develop an UI that will help in the refinement process. The user preferences will be provided and the algorithm will incorporate the new knowledge provided. Moreover, this interface will also be able to initiate this process by asking for potentially important information from the user.


The data set used for our case study is Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX from https://www.unb.ca/cic/datasets/ids-2017.html
This data was first preprossessed with the 2 python scripts in /Machine-Guided-Interactive-Clustering/datasets/

# USER MANUAL
To run this project, you will need to open two terminals. One to run the server, and another for the frontend.

Terminal 1: run the server
1. navigate to .../Machine-Guided-Interactive-Clustering
2. run "npm install"
3. run "npm start server"
4. you should see a message similar to "server is running at port 4500"

Terminal 2: run the frontend
1. navigate to .../Machine-Guided-Interactive-Clustering/interactive-constrained-clustering
2. run "npm install"
3. run "npm start"
4. the project should automatically open in your browser
