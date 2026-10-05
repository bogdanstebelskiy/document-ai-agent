# Docker

A Docker image is a packaged filesystem and configuration used to create containers. A container is a running instance of an image with its own isolated process environment.

A Dockerfile describes how an image is built. Each instruction can contribute a layer to the resulting image. Keeping dependencies and frequently changing files in sensible parts of the Dockerfile can improve build caching.

A container's writable filesystem is ephemeral. A volume is useful when data needs to survive container replacement.

Docker Compose is convenient when an application needs several local services, such as an API, database, and Redis. Services can communicate using the service names defined in the Compose file.
