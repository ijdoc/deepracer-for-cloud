FROM ubuntu:latest

# Install SSH client and other necessary utilities
RUN apt-get update && apt-get install -y ssh-client

# Add your SSH private key (ensure this is done securely)
COPY id_ed25519 /root/.ssh/id_ed25519
RUN chmod 600 /root/.ssh/id_ed25519
RUN echo "Host *\n\tStrictHostKeyChecking no\n" > /root/.ssh/config

# Set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]