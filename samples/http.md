# HTTP

HTTP is a request-response protocol. A client sends a request and a server returns a response. A request commonly contains a method, target URL, headers, and optionally a body.

GET is normally used to retrieve a resource, while POST commonly asks the server to process or create something. PUT is generally used to replace a resource, and PATCH is commonly used for partial updates.

A 2xx response indicates success. A 4xx response generally means the request could not be fulfilled because of something about the client request, while 5xx indicates a server-side failure.

HTTP headers carry metadata such as content type, authorization information, caching instructions, and cookies. Status codes and headers are separate parts of the response.
