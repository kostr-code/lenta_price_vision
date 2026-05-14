FROM node:20-alpine AS build

WORKDIR /app

COPY packages/frontend/package.json packages/frontend/package-lock.json ./

RUN npm ci

COPY packages/frontend/ ./

ARG VITE_BACKEND_URL=http://localhost:8001
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

RUN npm run build

FROM nginx:1.27-alpine

COPY infra/docker/frontend.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
