# Use Node.js base image
FROM node:18

# Set the working directory in the container
WORKDIR /usr/src/app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy the app code
COPY . .

# Expose the port that the app will run on
EXPOSE 8080

# Set the environment variable for the port (use 8080 since Cloud Run uses this port)
ENV PORT=8080

# Start the app
CMD ["node", "index.js"]