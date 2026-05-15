FROM node:20-alpine

WORKDIR /app

COPY packages/frontend/package.json packages/frontend/package-lock.json ./

RUN npm ci

COPY packages/frontend/ ./

ARG VITE_BACKEND_URL=http://localhost:8001
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

RUN npm run build

EXPOSE 4173

CMD ["npm", "run", "preview"]
