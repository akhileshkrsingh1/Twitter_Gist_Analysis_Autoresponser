$(".hideText").show();
$(".show-text").hide();

$(".arrow-next").click(function() {
    $(".hideText").hide();
    $(".show-text").show();
});

$(".arrow-show").click(function() {
    $(".hideText").show();
    $(".show-text").hide();
});

var arrows = document.querySelectorAll(".arrow-main");

arrows.forEach(function(arrow) {
    arrow.addEventListener("click", function(e) {
        e.preventDefault();

        if (!arrow.classList.contains("animate")) {
            arrow.classList.add("animate");
            setTimeout(function() {
                arrow.classList.remove("animate");
            }, 1600);
        }
    });
});

let currentTweetIndex = 0;
let tweets = [];
let currentResponseIndex = 0;
let responses = [];

function fetchTweets() {
    try {
        $.ajax({
            type: 'GET',
            url: 'http://localhost:5000/tweets',
            success: function(response) {
                tweets = response.tweets;
                if (tweets.length > 0) {
                    displayTweet(0);
                }
            },
            error: function(error) {
                console.error("AJAX request failed:", error);
                alert('Error fetching tweets: ' + (error.responseJSON ? error.responseJSON.error : error.statusText));
            }
        });
    } catch (error) {
        console.error("Error in fetchTweets:", error);
        alert('An unexpected error occurred while fetching tweets.');
    }
}

function displayTweet(index) {
    const tweet = tweets[index];
    $('#username').text(tweet.username);
    $('#tweet-content').text(tweet.text);
    
    //Image display
    const imagePath = `img/${tweet.image}`;
    if (tweet.image === null) {
        $('#tweet-image').attr('src', 'img/default.png');
    }
    else {
    $('#tweet-image').attr('src', imagePath);
    }
    // Fetch and display sentiment, categories, and response from the drafted reply
    try {
        $.ajax({
            type: 'GET',
            url: `http://localhost:5000/draft/${tweet.tweet_id}`,
            success: function(response) {
                console.log("Tweet priority:", response.priority);
                console.log("Tweet engagement:", tweet.profile);
                // Determine the priority class
                let priorityClass = '';
                let engagementClass = '';
                if (tweet.profile === 'High Engagement') {
                    engagementClass = 'high-engagement';
                } else if (tweet.profile === 'Low Engagement') {
                    engagementClass = 'low-engagement';
                }
                if (response.priority === 'High Priority') {
                    priorityClass = 'high-priority';
                } else if (response.priority === 'Low Priority') {
                    priorityClass = 'low-priority';
                }
                // Update the priority text and class
                $('#priority-text').text(response.priority);
                $('#engagement-text').text(tweet.profile);
                $('.circle').removeClass('high-priority-c low-priority-c').addClass(priorityClass+'-c');
                $('.ecircle').removeClass('high-engagement-c low-engagement-c').addClass(engagementClass+'-c');
                $('#priority-text').removeClass('high-priority low-priority').addClass(priorityClass);
                $('#engagement-text').removeClass('high-engagement low-engagement').addClass(engagementClass);


                // Update sentiment
                $('#sentiment-positive').removeClass('active-green');
                $('#sentiment-negative').removeClass('active');
                $('#sentiment-neutral').removeClass('active');

                if (response.sentiment) {
                    if (response.sentiment.toLowerCase() === 'positive') {
                        $('#sentiment-positive').addClass('active-green');
                    } else if (response.sentiment.toLowerCase() === 'negative') {
                        $('#sentiment-negative').addClass('active');
                    } else if (response.sentiment.toLowerCase() === 'neutral') {
                        $('#sentiment-neutral').addClass('active');
                    }
                }

                // Update categories
                $('#gist-categories').empty();
                response.categories.forEach(function(category) {
                    $('#gist-categories').append('<button class="enable-button"><span class="positive active-blue">' + category + '</span></button>');
                });

                // Update response
                responses = response.draft_reply[0];
                currentResponseIndex = 0;
                displayResponse(currentResponseIndex);
            },
            error: function(error) {
                console.error("AJAX request failed:", error);
                alert('Error fetching draft reply: ' + (error.responseJSON ? error.responseJSON.error : error.statusText));
            }
        });
    } catch (error) {
        console.error("Error in displayTweet:", error);
        alert('An unexpected error occurred while displaying the tweet.');
    }
}

function displayResponse(index) {
    try {
        if (responses.length > 0) {
            const response = responses[index];
            $('#response-text').text(response);  // Display the response text
            console.log("Response text:", response);
            $('#response-index').text(`${index + 1}/${responses.length}`);
        }
    } catch (error) {
        console.error("Error in displayResponse:", error);
        alert('An unexpected error occurred while displaying the response.');
    }
}

function navigateResponse(direction) {
    try {
        if (direction === 'next') {
            if (currentResponseIndex < responses.length - 1) {
                currentResponseIndex++;
            }
        } else if (direction === 'prev') {
            if (currentResponseIndex > 0) {
                currentResponseIndex--;
            }
        }
        console.log("Current response index:", currentResponseIndex);
        displayResponse(currentResponseIndex);
    } catch (error) {
        console.error("Error in navigateResponse:", error);
        alert('An unexpected error occurred while navigating responses.');
    }
}

function navigateTweet(direction) {
    try {
        if (direction === 'next') {
            if (currentTweetIndex < tweets.length - 1) {
                currentTweetIndex++;
            }
        } else if (direction === 'prev') {
            if (currentTweetIndex > 0) {
                currentTweetIndex--;
            }
        }
        displayTweet(currentTweetIndex);
    } catch (error) {
        console.error("Error in navigateTweet:", error);
        alert('An unexpected error occurred while navigating tweets.');
    }
}

function uploadTweets() {
    const fileInput = document.getElementById('uploadFile');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(event) {
        const tweets = JSON.parse(event.target.result);
        $('#loader').show();
      
        try {
            $.ajax({
                type: 'POST',
                url: 'http://localhost:5000/upload',
                data: JSON.stringify(tweets),
                contentType: 'application/json; charset=utf-8',
                success: function(response) {
                    alert('Tweets uploaded successfully.');
                    $('#loader').hide();
                    fetchTweets();  // Refresh the tweets list after uploading
                },
                error: function(error) {
                    console.error("AJAX request failed:", error);
                    alert('Error uploading tweets: ' + (error.responseJSON ? error.responseJSON.error : error.statusText));
                    $('#loader').hide();
                }
            });
        } catch (error) {
            console.error("Error in uploadTweets:", error);
            alert('An unexpected error occurred while uploading tweets.');
            $('#loader').hide();
        }
    };
    reader.readAsText(file);
}
