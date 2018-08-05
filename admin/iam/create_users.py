import boto3
import csv
import random
import string

people = []
with open('fsdl.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        people.append(row)

session = boto3.Session(
)

data = []

for person in people:
    first, last, email = person
    iam = session.resource('iam')
    user = iam.User(email)
    #user.remove_group(GroupName='students')
    #for access_key in user.access_keys.all():
    #    access_key.delete()
    #user.delete()
    user.create()
    user.add_group(GroupName='students')
    access_key_pair = user.create_access_key_pair()
    password = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    sls = 'sls config credentials --provider aws --key {0} --secret {1}'.format(access_key_pair.access_key_id, access_key_pair.secret_access_key)
    print(sls)
    datum = [first, last, email, access_key_pair.access_key_id, access_key_pair.secret_access_key, password, sls]
    login_profile = user.create_login_profile(
        Password=password
    )
    data.append(datum)

print(data)

with open('fsdl_email.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    for datum in data:
        spamwriter.writerow(datum)
