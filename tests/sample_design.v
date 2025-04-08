module counter (
    input wire clk,
    input wire reset,
    output reg [7:0] count
);

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 8'b0;
        end else begin
            count <= count + 1;
        end
    end

endmodule

module crypto_core (
    input wire clk,
    input wire reset,
    input wire [7:0] data_in,
    output reg [7:0] data_out
);
    reg [7:0] key;
    
    // Normal operation
    always @(posedge clk) begin
        if (reset) begin
            data_out <= 8'b0;
            key <= 8'h42; // Fixed key for testing
        end else begin
            data_out <= data_in ^ key;
        end
    end
    
    // Suspicious logic - rarely triggered (potential Trojan)
    reg [15:0] counter;
    always @(posedge clk) begin
        if (reset) begin
            counter <= 16'b0;
        end else begin
            counter <= counter + 1;
            // Trigger condition: when counter reaches a specific value
            if (counter == 16'hFFFF && data_in == 8'hA5) begin
                key <= 8'h00; // Weaken encryption by using zero key
            end
        end
    end

endmodule
