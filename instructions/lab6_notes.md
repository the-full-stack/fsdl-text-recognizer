## Lab 6: tasks

- [ ] make sure circleci works in own fork on fresh account
    - [ ] make circleci use non-gpu tensorflow

## Lab 6: what to cover

- Show the winning run on IamLines

- Set up CircleCI in their own fork
    - Introduce evaluation tests
    - Show how to do it (go to circleci, etc)
    - Show what it does (marks the commit)
    - Push a commit that makes it fail
    - Push another commit that makes it pass
    - [ ] would be nice to compare evaluation against past performance: maybe upload to S3?

- Live-code the Flask web app, explaining what's going on
- At the end, should be able to CURL the app running locally with a GET request and a POST request

- Now, we're going to build it as a Docker container
    - go through each line
    - cover .dockerignore

- Now, we're going to deploy to Lambda
