import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl
data = [
    {
        "resume_text": "Experienced Software Engineer with 3+ years of backend development using Python and Flask. Designed and deployed scalable RESTful APIs integrated with PostgreSQL databases. Led the migration of monolithic applications to microservices using Docker and AWS ECS. Familiar with CI/CD pipelines, Git workflows, and agile development. Collaborated cross-functionally with frontend teams and DevOps. Deployed applications on AWS EC2 and S3, and managed infrastructure using Terraform.",
        "job_description": "We are looking for a Backend Software Engineer proficient in Python and Flask to build and maintain scalable APIs. Responsibilities include designing microservices, working with PostgreSQL, deploying via Docker containers, and managing AWS services. Candidates should be comfortable with version control, agile development, and CI/CD pipelines. Experience with cloud infrastructure and RESTful API design is essential.",
        "label": 1
    },
    {
        "resume_text": "High school English teacher with 7 years of classroom experience. Specialized in modern literature, creative writing workshops, and SAT prep. Developed interdisciplinary lesson plans and managed student writing portfolios. Coordinated school-wide reading initiatives and led after-school tutoring programs.",
        "job_description": "We are hiring a Data Engineer to develop data ingestion pipelines, manage Spark jobs, and optimize ETL workflows. Applicants should have experience with big data frameworks, Python or Scala, and cloud platforms like AWS or GCP. Familiarity with Airflow and relational databases is a plus.",
        "label": 0
    },
    {
        "resume_text": "Computer Science undergraduate with internship experience at a fintech startup. Developed React-based dashboards to visualize financial metrics and implemented secure API integrations. Proficient in JavaScript, React, REST APIs, and Git. Built frontend components in Agile sprints and conducted usability testing to improve user experience.",
        "job_description": "We are seeking a front-end developer to join our product team. Candidates must be skilled in React and have experience building interactive web applications. Understanding of state management (e.g. Redux), RESTful integration, and component testing is preferred.",
        "label": 1
    },
    {
        "resume_text": "Barista and assistant manager with 4 years of experience in customer-facing roles. Supervised staff schedules, managed supply orders, and ensured customer satisfaction in a high-volume café environment. Recognized for team leadership and process improvements that reduced wait times by 20%.",
        "job_description": "Join our tech startup as a full-stack engineer responsible for designing web apps, integrating backend APIs, and maintaining CI/CD workflows. Experience with JavaScript frameworks, Python backends, and cloud deployments is expected.",
        "label": 0
    },
    {
        "resume_text": "Motivated Computer Science student with strong academic background in AI and machine learning. Completed several academic and personal projects using TensorFlow and scikit-learn. Built CNN models for image classification and NLP models for sentiment analysis. Experience with Jupyter notebooks, data preprocessing, and model evaluation techniques.",
        "job_description": "We are hiring a machine learning intern with familiarity in TensorFlow, Python, and data science workflows. You will support the development of classification models and contribute to preprocessing pipelines and experimentation logging.",
        "label": 1
    },
    {
        "resume_text": "Digital marketing specialist with 5+ years of experience in managing social media ad campaigns, SEO optimization, and influencer partnerships. Grew client engagement by 35% using targeted ad strategies. Proficient in Google Analytics, Meta Ads Manager, and Canva.",
        "job_description": "Software engineer needed to design backend systems for our B2B SaaS platform. Required skills include API development, data modeling, and cloud-based service design using Python or Go. Bonus for experience with container orchestration (e.g. Kubernetes).",
        "label": 0
    },
    {
        "resume_text": "Backend engineer with 6 years of experience in building distributed systems using Go. Designed scalable microservices, implemented gRPC APIs, and maintained Kubernetes deployments. Familiar with CI/CD tools like Jenkins and observability stacks like Prometheus and Grafana.",
        "job_description": "Looking for backend engineers with strong Go development skills. You will build robust APIs, manage cloud-native services, and work closely with our DevOps team. Experience with containerization and monitoring tools is desirable.",
        "label": 1
    },
    {
        "resume_text": "Biology graduate with lab research experience in genetics and cell imaging. Conducted PCR, Western blotting, and microscopy experiments. Authored a senior thesis on epigenetic factors in gene expression. Seeking entry-level roles in biotech or healthcare.",
        "job_description": "Looking for a junior software developer to support internal tool development. Must be familiar with Java or Python and have solid understanding of algorithms, data structures, and software engineering principles.",
        "label": 0
    },
    {
        "resume_text": "Full-stack developer with 4 years of experience building MERN stack applications. Designed REST APIs using Express and built responsive UIs in React. Proficient in MongoDB, GitHub Actions, Docker, and deployment on Heroku and AWS. Team player with strong Agile experience.",
        "job_description": "Hiring a full-stack JavaScript developer experienced in React and Node.js. You'll help maintain our web platform and collaborate with design and QA teams. Familiarity with NoSQL databases and CI/CD tools is a must.",
        "label": 1
    },
    {
        "resume_text": "Retail associate with 3+ years of experience in customer service, inventory management, and point-of-sale systems. Consistently exceeded sales targets and trained new staff. Awarded Employee of the Month three times for outstanding service.",
        "job_description": "Customer support associate wanted for a growing e-commerce platform. The ideal candidate has experience with helpdesk tools, resolving customer inquiries, and maintaining a friendly, professional tone across all channels.",
        "label": 1
    }
] #This is temporary, when it is larger, make it a json file and move it

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed_model(data_values):
    return embed(data_values).numpy()
resumes = [d["resume_text"] for  d in data]
jobs = [d["job_description"] for  d in data]
labels = [d["label"] for  d in data]
resume_vectors = embed_model(resumes)
job_vectors = embed_model(jobs)
cosine_sims = [cosine_similarity([resume_vectors[x]], [job_vectors[x]])[0][0] for x in range(len(resume_vectors))]
weight_value = 0.1 #Can be changed
cosine_sims = np.array(cosine_sims).reshape(-1, 1) * weight_value
x = np.concatenate([resume_vectors, job_vectors, cosine_sims], axis = 1)
y = np.array(labels).astype(np.float32)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation = "relu", input_shape = (x.shape[1], )),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 10, batch_size = 2)
model.save("app/model/temp_model.h5")
with open("app/model/temp_data.pkl", "wb") as f:
    pkl.dump({
        "resume_vectors": resume_vectors,
        "job_vectors": job_vectors,
        "cosine_sims": cosine_sims,
        "x": x,
        "y": y
    }, f)