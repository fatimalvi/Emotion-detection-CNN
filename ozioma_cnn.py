# Define the model based on the provided architecture and hyperparameters
model = Sequential([
    # Rescaling layer to normalize the input images
    # Rescaling(1./255, input_shape=(48, 48, 1)),

    # Convolutional layers
    Conv2D(48, kernel_size=(3, 3), activation='relu'),
    Conv2D(48, kernel_size=(3, 3), activation='relu'),

    # MaxPooling and Dropout layers
    MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    Dropout(0.5),

    # More Convolutional layers
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),

    # MaxPooling and Dropout layers
    MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    # Even more Convolutional layers
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    Conv2D(256, kernel_size=(3, 3), activation='relu'),

    # MaxPooling and Dropout layers
    MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    Dropout(0.5),

    # Flatten the output to feed into the dense layers
    Flatten(),

    # Dense layers with dropout
    Dense(1024, activation='relu'),
    Dropout(0.5),

    # Output layer with 7 classes for FER2013 dataset
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # Use categorical_crossentropy instead of sparse_categorical_crossentropy
    metrics=['accuracy']
)
