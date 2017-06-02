from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors

# Store your full project ID in a variable in the format the API needs.
projectID = 'projects/{}'.format('cloudml-demo')

# Get application default credentials (possible only if the gcloud tool is
#  configured on your machine).
# gcloud auth application-default login
credentials = GoogleCredentials.get_application_default()

# Build a representation of the Cloud ML API.
ml = discovery.build('ml', 'v1', credentials=credentials)

# Create a dictionary with the fields from the request body.
# requestDict = {'name': 'api_model1', 'description': 'a model from the python api'}

# Create a request to call projects.models.list.
request = ml.projects().models().list(
                      parent=projectID) #, body=requestDict)

# Make the call.
try:
    response = request.execute()
    print(response)

except errors.HttpError as err:
    # Something went wrong, print out some information.
    print('There was an error creating the model. Check the details:')
    print(err._get_reason())
    print(err)
