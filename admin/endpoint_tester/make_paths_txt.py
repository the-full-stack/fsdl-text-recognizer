with open('endpoints.txt') as endpoints_file:
    endpoints = [x.strip() for x in endpoints_file.readlines()]
with open('remote_images.txt') as image_urls_file:
    remote_image_urls = [x.strip() for x in image_urls_file.readlines()]

paths = []
for endpoint in endpoints:
    for rem in remote_image_urls:
        s = '{0}/v1/predict?image_url={1}'.format(endpoint, rem)
        paths.append(s)
        print s
