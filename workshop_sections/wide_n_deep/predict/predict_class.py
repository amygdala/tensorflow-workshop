from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
import json


class MLEngine:
	def __init__(self, projectID='cloudml-demo', service='ml', version='v1'):
		self.projectID = projectID
		self.service=service
		self.version=version
		self.svc = self.make_svc()

	def make_svc(self):
		# Get application default credentials (possible only if the gcloud tool is
		#  configured on your machine).
		# gcloud auth application-default login
		credentials = GoogleCredentials.get_application_default()

		# Build a representation of the Cloud ML API.
		ml = discovery.build(self.service, self.version, credentials=credentials)

		return ml

	def models_list(self):
		print('models.list')
		request = self.svc.projects().models().list(
		                      parent='projects/{}'.format(self.projectID)) #, body=requestDict)

		# Make the call.
		try:
		    response = request.execute()
		    print(response)

		except errors.HttpError as err:
		    # Something went wrong, print out some information.
		    print('There was an error listing the model. Details:')
		    print(err._get_reason())
		    print(err)

	def model_predict(self, model, version):
		print('models.predict')
		instances = []
		model_id = 'projects/{}/models/{}/versions/{}'.format(self.projectID, model, version)
		model_id = 'projects/{}/models/{}'.format(self.projectID, model)

		print(model_id)

		with open('test.json') as infile:
			for line in infile:
				instances.append(json.loads(line))

		request_body = {'instances': instances}

		request = self.svc.projects().predict(
		                      # parent=self.projectID,
		                      name=model_id,
		                      body=request_body
		                      ) #, body=requestDict)

		# Make the call.
		try:
		    response = request.execute()
		    print(response)

		except errors.HttpError as err:
		    # Something went wrong, print out some information.
		    print('There was an error listing the model. Details:')
		    print(err._get_reason())
		    print(err)


def make_models():
	ml = MLEngine()

	# ml.models_list()

	ml.model_predict('cloudwnd', 'v1')

	return

if __name__ == "__main__":
	make_models()

# Create a dictionary with the fields from the request body.
# requestDict = {'name': 'api_model1', 'description': 'a model from the python api'}

# Create a request to call projects.models.list.
# request = ml.svc.projects().models().list(
#                       parent=ml.projectID) #, body=requestDict)

# # Make the call.
# try:
#     response = request.execute()
#     print(response)

# except errors.HttpError as err:
#     # Something went wrong, print out some information.
#     print('There was an error creating the model. Check the details:')
#     print(err._get_reason())
#     print(err)
