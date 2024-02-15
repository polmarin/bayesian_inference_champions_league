library("rjags")
library(ggplot2)

# Load the data
con <- dbConnect(duckdb(), dbdir = "/Users/polmarin/Documents/Coding/Python/SportsAnalytics/football_data.db", read_only = TRUE)
data <- dbGetQuery(
  con, 
  "WITH home_t AS (
    SELECT 
      start_time,
      ROUND(h_s.avg_player_rating) AS avg_rating,
      CAST(split_part(m.score, '-', 1) AS INTEGER) AS goals,
      1 AS home,
      CAST(split_part(m.score, '-', 1) > split_part(m.score, '-', 2) AS INTEGER) AS win
      FROM match_info AS m 
      INNER JOIN team_stats AS h_s ON (m.home_team_id = h_s.team_id AND m.id = h_s.match_id)
      INNER JOIN teams AS h_t ON (m.home_team_id = h_t.id)
      WHERE league_id = 12 AND season = '2023/2024'
  ), 
  away_t AS (
    SELECT 
      start_time,
      ROUND(a_s.avg_player_rating) AS avg_rating,
      CAST(split_part(m.score, '-', 2) AS INTEGER) AS goals,
      0 AS home,
      CAST(split_part(m.score, '-', 1) < split_part(m.score, '-', 2) AS INTEGER) AS win
      FROM match_info AS m 
      INNER JOIN team_stats AS a_s ON (m.away_team_id = a_s.team_id AND m.id = a_s.match_id)
      INNER JOIN teams AS a_t ON (m.away_team_id = a_t.id)
      WHERE league_id = 12 AND season = '2023/2024'
  )
  SELECT 
    goals, home AS is_home, avg_rating, win
  FROM (
    SELECT * FROM home_t
    UNION
    SELECT * FROM away_t)
  ORDER BY start_time DESC
  "
)

# Create JAGS model
modelString = "
model{
  # Likelihood
  for(i in 1:n){
    y[i] ~ dbern(pi[i])
    logit(pi[i]) <- beta[1] + beta[2]*goals[i] + beta[3]*is_home[i] + beta[4]*avg_rating[i]
  }
  
  # Prior distributions
  for (j in 1:4)
  {
    beta[j] ~ dnorm(0.0, 0.0001)
  }
}"
writeLines( modelString , con="cl_model.txt" )

# Initialize values
init_values <- function(){
  list(beta = rnorm(4, 0, 3))
}
init_values()

# Prepare JAGS data
jags_data <- list(y = data$win,
                  is_home = data$is_home,
                  goals = data$goals,
                  #GC = data$GC,
                  avg_rating = data$avg_rating,
                  n = nrow(data))

# Model's setup and adaptation period
jagsModel <- jags.model(file = "cl_model.txt",
                        data = jags_data,
                        inits = init_values,
                        n.chains = 3,
                        n.adapt = 300)

# Model's burn-in
update(jagsModel, n.iter=1000)

# Predict MCMC samples
samples <- coda.samples(jagsModel,
                            variable.names = c("beta"),
                            n.iter = 10000,
                            thin = 10)

# Analyze the chains
diagMCMC(codaObject = samples, parName = "beta[2]")

# Plot final distributions
plotPost( samples[,"beta[2]"], xlab="beta[2]")


# Predict game using last 5 game stats for each team
avg_data <- dbGetQuery(
  con, 
  "WITH home_t AS (
    SELECT 
      start_time,
      h_t.id AS team_id,
      h_t.name AS team,
      ROUND(h_s.avg_player_rating) AS avg_rating,
      CAST(split_part(m.score, '-', 1) AS INTEGER) AS goals,
      1 AS home
      FROM match_info AS m 
      INNER JOIN team_stats AS h_s ON (m.home_team_id = h_s.team_id AND m.id = h_s.match_id)
      INNER JOIN teams AS h_t ON (m.home_team_id = h_t.id)
      WHERE season = '2023/2024'
  ), 
  away_t AS (
    SELECT 
      start_time,
      a_t.id AS team_id,
      a_t.name AS team,
      ROUND(a_s.avg_player_rating) AS avg_rating,
      CAST(split_part(m.score, '-', 2) AS INTEGER) AS goals,
      0 AS home
      FROM match_info AS m 
      INNER JOIN team_stats AS a_s ON (m.away_team_id = a_s.team_id AND m.id = a_s.match_id)
      INNER JOIN teams AS a_t ON (m.away_team_id = a_t.id)
      WHERE season = '2023/2024'
  ), unioned AS (
    SELECT 
      team_id, team, goals, avg_rating,
      ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY start_time DESC) AS r
    FROM (
      SELECT * FROM home_t
      UNION
      SELECT * FROM away_t)
  )
  
  SELECT 
    team_id, team,
    AVG(goals) AS goals_avg,
    AVG(avg_rating) AS rating_avg
  FROM unioned
  WHERE r <= 5
  GROUP BY 1,2
  ORDER BY 1
  "
)

# Home team data
home_team = 'Napoli'
home_pred_home <- 1
goals_pred_home <- avg_data[avg_data$team == home_team, 'goals_avg']
avg_rating_pred_home <- avg_data[avg_data$team == home_team, 'rating_avg']

# Away team data
away_team = 'Barcelona'
home_pred_away <- 0 
goals_pred_away <- avg_data[avg_data$team == away_team, 'goals_avg']
avg_rating_pred_away <- avg_data[avg_data$team == away_team, 'rating_avg']

# Compute predictors
predictor_pred_home <- samples[[1]][,1] +
  samples[[1]][,2]*goals_pred_home +
  samples[[1]][,3]*home_pred_home  +
  samples[[1]][,4]*avg_rating_pred_home
predictor_pred_away <- samples[[1]][,1] +
  samples[[1]][,2]*goals_pred_away +
  samples[[1]][,3]*home_pred_away  +
  samples[[1]][,4]*avg_rating_pred_away

# Compute pi (inverse of logit)
pi_pred_home <- as.numeric(exp(predictor_pred_home)/(1+exp(predictor_pred_home)))
pi_pred_away <- as.numeric(exp(predictor_pred_away)/(1+exp(predictor_pred_away)))

# Plot the results
preds <- data.frame(pi_pred_home, pi_pred_away)
names(preds) <- c(home_team, away_team)

ggplot(preds) +
  geom_histogram(aes(x = pi_pred_home, y = ..density.., color = home_team, fill=home_team),
                 bins = 30, alpha=0.5) +
  geom_histogram(aes(x = pi_pred_away, y = ..density.., color = away_team, fill=away_team ),
                 bins = 30, alpha=0.5) +
  theme_light() +
  xlim(c(0,1)) +
  xlab(expression(pi)) +
  theme(axis.text = element_text(size=15),
        axis.title = element_text(size=16))+
  ggtitle(paste(home_team, 'vs', away_team))

