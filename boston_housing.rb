require 'matrix'
require 'rumale'
require 'daru'
require 'http'

# Download the dataset
def download_dataset(url)
  response = HTTP.get(url)
  if response.status.success?
    File.open('BostonHousing.csv', 'w') { |file| file.write(response.to_s) }
  else
    puts "Error downloading the dataset."
    exit
  end
end

# Load the dataset
def load_dataset(file_path)
  Daru::DataFrame.from_csv(file_path)
end

# Prepare the data
def prepare_data(data_frame)
  y = data_frame['medv'].to_a
  x = data_frame.delete_vector('medv')
  [x, y]
end

# Main
dataset_url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
dataset_file_path = 'BostonHousing.csv'

download_dataset(dataset_url) unless File.exist?(dataset_file_path)
data_frame = load_dataset(dataset_file_path)
x, y = prepare_data(data_frame)

n_samples = data_frame.nrows

# Split the data into training and testing sets
splitter = Rumale::ModelSelection::ShuffleSplit.new(n_splits: 1, test_size: 0.2, random_seed: 43)
train_ids, test_ids = splitter.split(x, y).first

x_train = x.row[*train_ids].to_matrix
x_test = x.row[*test_ids].to_matrix
y_train = y.values_at(*train_ids)
y_test = Numo::DFloat.cast(y.values_at(*test_ids))

# Train the model
model = Rumale::LinearModel::Ridge.new(reg_param: 0)
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Evaluate the model
mae = Rumale::EvaluationMeasure::MeanAbsoluteError.new.score(y_test, predictions)
puts "Mean Absolute Error: #{mae}"

# Calculate Mean Squared Error (MSE)
mse = Rumale::EvaluationMeasure::MeanSquaredError.new
mse_score = mse.score(y_test, predictions)

# Calculate R-squared score
r2 = Rumale::EvaluationMeasure::R2Score.new
r2_score = r2.score(y_test, predictions)

puts "Mean Squared Error: #{mse_score}"
puts "R-squared Score: #{r2_score}"
