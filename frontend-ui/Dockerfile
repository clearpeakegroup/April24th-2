# syntax=docker/dockerfile:1.4
FROM node:20

WORKDIR /app

COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"] 