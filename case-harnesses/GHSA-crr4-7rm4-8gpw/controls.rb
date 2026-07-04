# frozen_string_literal: true

require 'ipaddr'

load '/workspace/app/lib/private_address_check.rb'

expectations = {
  '127.0.0.1' => true,
  '10.0.0.1' => true,
  '169.254.1.2' => true,
  '8.8.8.8' => false,
  '2001:4860:4860::8888' => false,
}

failures = expectations.filter_map do |text, expected|
  actual = PrivateAddressCheck.private_address?(IPAddr.new(text))
  "#{text}: expected #{expected}, got #{actual}" unless actual == expected
end

unless failures.empty?
  warn failures.join("\n")
  exit 1
end

puts "#{expectations.length} compatibility controls passed"
