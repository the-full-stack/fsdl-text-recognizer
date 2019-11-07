# Lab 9: Monitoring running web service

## Monitoring

We can look at the requests our function is receiving in the AWS CloudWatch interface.
It shows requests, errors, duration, and some other metrics.

What it does not show is stuff that we care about specifically regarding machine learning: data and prediction distributions.

This is why we added a few extra metrics to `api/app.py`, in `predict()`.
Using these simple print statements, we can set up CloudWatch metrics by using the Log Metrics functionality.

### Log Metrics

Log in to your AWS Console, and make sure you're in the `us-west-2` region.

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
