Note that emailing credentials is a bad idea. You usually want to handle credentials in a more secure fashion.
We're only doing it in this case because your credentials give you limited access and are for a temporary AWS account.

You can also go to https://379872101858.signin.aws.amazon.com/console and log in with the email you used to register (and the password we emailed you), and create your own credentials if you prefer.

## Lambda monitoring

We're going to check the logs and set up monitoring for your deployed API. In order to make the monitoring more interesting, we're going to simulate people using your API.

**In order for us to do that, you need to go to https://goo.gl/forms/YQCXTI2k5R5Stq3u2 and submit your endpoint URL.**
It should look like this (ending in "/dev/"):
```
https://REPLACE_THIS.execute-api.us-west-2.amazonaws.com/dev/
```

If you haven't already sent a few requests to your endpoint, you should do so using the curl commands above.

Next, log in to the AWS Console at https://379872101858.signin.aws.amazon.com/console (you should've gotten an email with your username and password).

**Make sure that you switch into the Oregon region (also known as `us-west-2`) using the dropdown menu in the top right corner.**

Once you're in, click on 'Services' and go to 'CloudWatch' under 'Management Tools.' Click on 'Logs' in the left sidebar. This will have several log groups -- one for each of us.
You can filter for yours by entering `/aws/lambda/text-recognizer-USERNAME-dev-api` (you need to enter the whole thing, not just your username).
Click on yours. You'll some log streams. If you click on one, you'll see some logs for requests to your API. Each log entry starts with START and ends with REPORT. The REPORT line has some interesting information about the API call, including memory usage and duration.

We're also logging a couple of metrics for you: the confidences of the predictor and the mean intensities of the input images.
Next, we're going to make it so you can visualize these metrics. Go back to the list of Log Groups by clicking on Logs again in the left sidebar.
Find your log group, but don't click on it. You'll see a column that says 'Metric Filters.' You currently likely have 0 filters. Click on "0 filters."
Click on 'Add Metric Filter.'

Now, we need to add a pattern for parsing our metric out of the logs. Here's one you can use for the confidence levels. Enter this in the 'Filter Pattern' box.
```
[level=METRIC, metric_name=confidence, metric_value]
```
Click on 'Assign Metric.'
Now, we need to name the metric and tell it what the data source is. Enter 'USERNAME_confidence' in the 'Metric name' box (replace USERNAME as usual). Click on 'Show advanced metric settings,' and for Metric Value, click on $metric_value to populate the text box. Hit 'Create Filter.'
Since we're already here, let's go ahead and make another metric filter for the mean intensity. You can use this Filter Pattern:
```
[level=METRIC, metric_name=mean_intensity, metric_value]
```
You should name your metric "USERNAME_mean_intensity."

Now we have a couple of metric filters set up.
Unfortunately, Metric Filters only apply to new log entries, so go back to your terminal and send a few more requests to your endpoint.

Now we can make a dashboard that shows our metrics. Click on 'Dashboards' in the left sidebar. Click 'Create Dashboard.' Name your dashboard your USERNAME.

We're going to add a few widgets to your dashboard. For the first widget, select 'Line'. In the search box, search for your username.
Click on 'Lambda > By Function Name' in the search results, and select the checkbox for 'Invocations.' This'll make a plot showing you much your API is being called.

Let's add another widget -- select Line again. Go back to the Lambda metrics and select 'Duration' this time.

Lastly, let's plot our custom metrics. Add one more 'Line' widget, search for your username again, and click on 'LogMetrics' and then 'Metrics with no dimensions'.
Check two checkboxes: `USERNAME_confidence` and `USERNAME_mean_intensity.` Before hitting Create, click on the 'Graphed Metrics' tab above, and under the 'Y Axis' column,
select the right arrow for one of the metrics (it doesn't matter which one). Now hit create.

Feel free to resize and reorder your widgets.

Make sure to save your dashboard -- else it won't persist across sessions.

You can play with your API here a bit while we turn on the traffic for everyone. Double check that you've submitted your endpoint to the Google form above.

Once the traffic is going, refresh your dashboard a bit and watch it. We're going to change something about the traffic, and it's going to start making your API perform poorly.
Try and figure out what's going on, and how you can fix it. We'll leave the adversarial traffic on for a while.

If you're curious, you can add a metric filter to show memory usage with this pattern:
```
[report_name="REPORT", request_id_name="RequestId:", request_id_value, duration_name="Duration:", duration_value, duration_unit="ms", billed_duration_name_1="Billed", bill_duration_name_2="Duration:", billed_duration_value, billed_duration_unit="ms", memory_size_name_1="Memory", memory_size_name_2="Size:", memory_size_value, memory_size_unit="MB", max_memory_used_name_1="Max", max_memory_used_name_2="Memory", max_memory_used_name_3="Used:", max_memory_used_value, max_memory_used_unit="MB"]
```

You can name it `USERNAME_memory`. Select `$max_memory_used_value` for the metric value.

Make sure to save your dashboard!
